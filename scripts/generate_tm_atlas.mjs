import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import initSqlJs from 'sql.js';
import { UMAP } from 'umap-js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const DB_PATH = path.join(REPO_ROOT, 'public', 'data', 'tm_misha_minilm.db');
const OUTPUT_PATH = path.join(REPO_ROOT, 'public', 'data', 'tm-atlas.json');
const SQL_WASM_PATH = path.join(REPO_ROOT, 'node_modules', 'sql.js', 'dist', 'sql-wasm.wasm');

const MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2';
const RANDOM_SEED = 24_031;
const UMAP_NEIGHBORS = 22;
const UMAP_MIN_DIST = 0.035;
const UMAP_SPREAD = 1.2;
const UMAP_EPOCHS = 220;
const UMAP_SUPERVISED_TARGET_WEIGHT = 0.01;
const SEMANTIC_CLUSTER_COUNT = 15;
const SEMANTIC_KMEANS_MAX_ITERATIONS = 30;
const SEMANTIC_KMEANS_SEED = RANDOM_SEED + 97;
const VISUAL_CLUSTER_CENTER_SPREAD = 0.58;
const VISUAL_INTRA_CLUSTER_SPREAD = 0.98;
const VISUAL_OUTLIER_SOFT_RADIUS = 155;
const VISUAL_OUTLIER_TAIL_SPREAD = 0.5;
const VISUAL_INITIAL_ZOOM = 1.36;
const PHRASE_COUNT = 8;
const COMPACTNESS_CORE_SHARE = 0.8;
const COMPACTNESS_REFERENCE_PERCENTILE = 0.9;

function wordSet(words) {
  return new Set(words.trim().split(/\s+/));
}

const STOPWORDS = wordSet(
  `
  a about actually after again all almost also am an and any anyway are around as at back be
  because been being bit but by can car cars come could day did do does doing done down drive
  driving even every first for from get getting go going good got had has have having he here
  how i if im in into is it its ive just kind know last let like little look lot made many me
  mean more most much my need no not now of off oh okay on one only or other our out over
  pretty probably quite really right said say see she should so some something still such
  than thank thanks that thats the their them then there these they thing think this those
  through time to too two up us very was way we well went were what when where which while who
  why will with would yeah yes you your
  arent cant couldnt didnt doesnt dont hasnt havent isnt shouldnt wasnt wont wouldnt youre
  `,
);
const LABEL_WEAK_TERMS = wordSet(
  `
  actually ahead alright alrighty already amazing awesome bad basically bcy bye cause chop ciao
  com cool course definitely drove driven else enjoy enjoyed euros everything exactly fantastic fine fun
  gonna goodbye good great guess guys happen happy hello hey hopefully hour insane kilometers
  kilos loved makes man maybe might nice nur ok okay people per please power pretty quite really
  reason sad sense stuff sure sweaters thing things totally want welcome wow year years yeah yep
  `,
);
const LABEL_CONCEPTS = [
  ['short replies', 'yes no okay ok alright alrighty sure exactly probably yep neutral'],
  ['confirmations', 'yes no okay ok alright alrighty sure exactly probably yep'],
  ['signoffs', 'bye ciao hello welcome goodbye watching'],
  ['thanks', 'thanks thank'],
  ['courtesy', 'please sorry welcome thank thanks'],
  ['plans', 'gonna next soon coming ahead hopefully hope start started happen happening'],
  ['timing', 'months season winter summer today tomorrow yesterday year years'],
  ['reasons', 'because cause reason reasons means why since'],
  ['explanations', 'because cause reason reasons means why since'],
  ['opinions', 'opinion honest important true fact overall reality impressions experience verdict dissatisfied'],
  ['praise', 'nice loved enjoyed cool fantastic amazing happy impressed liked beautiful epic good'],
  ['merch', 'products product licensed buy socks sweaters merch merchandise shop'],
  ['specs', 'horsepower kilometers kilos weight rpm temperature euros'],
  ['drivers', 'driver drivers drove driven taxi'],
  ['passengers', 'passenger passengers'],
].map(([label, terms]) => ({ label, terms: wordSet(terms) }));
const LABEL_CONCEPT_LABELS = new Set(LABEL_CONCEPTS.map((concept) => concept.label));

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function normalizeVector(values) {
  let magnitude = 0;
  for (const value of values) {
    magnitude += value * value;
  }
  magnitude = Math.sqrt(magnitude);
  return values.map((value) => value / magnitude);
}

function cosineDistance(left, right) {
  let dot = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
  }
  return Math.max(0, 1 - dot);
}

function percentile(sortedValues, share) {
  const index = Math.min(sortedValues.length - 1, Math.max(0, Math.round((sortedValues.length - 1) * share)));
  return sortedValues[index];
}

function scaleCoordinates(coordinates) {
  const dimensions = coordinates[0].length;
  const bounds = [];

  for (let dimension = 0; dimension < dimensions; dimension += 1) {
    const values = coordinates.map((coordinate) => coordinate[dimension]).sort((left, right) => left - right);
    const low = percentile(values, 0.01);
    const high = percentile(values, 0.99);
    bounds.push({ low, high, span: high - low || 1 });
  }

  return coordinates.map((coordinate) =>
    coordinate.map((value, dimension) => {
      const bound = bounds[dimension];
      const scaled = (value - bound.low) / bound.span;
      return Math.round(Math.min(1, Math.max(0, scaled)) * 1000);
    }),
  );
}

function runUmap(vectors, assignments) {
  const umap = new UMAP({
    nComponents: 3,
    nNeighbors: UMAP_NEIGHBORS,
    minDist: UMAP_MIN_DIST,
    spread: UMAP_SPREAD,
    nEpochs: UMAP_EPOCHS,
    distanceFn: cosineDistance,
    random: seededRandom(RANDOM_SEED + 1),
  });
  umap.setSupervisedProjection(assignments, { targetWeight: UMAP_SUPERVISED_TARGET_WEIGHT });
  return umap.fit(vectors);
}

function squaredDistance(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    const delta = left[index] - right[index];
    total += delta * delta;
  }
  return total;
}

function centerOfMembers(points, members) {
  const center = new Array(points[0].length).fill(0);
  for (const member of members) {
    const point = points[member];
    for (let dimension = 0; dimension < point.length; dimension += 1) {
      center[dimension] += point[dimension];
    }
  }
  return center.map((value) => value / members.length);
}

function dotProduct(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    total += left[index] * right[index];
  }
  return total;
}

function chooseWeightedIndex(weights, totalWeight, random) {
  let threshold = random() * totalWeight;
  for (let index = 0; index < weights.length; index += 1) {
    threshold -= weights[index];
    if (threshold <= 0) {
      return index;
    }
  }
  return weights.length - 1;
}

function initializeKMeansPlusPlus(vectors, clusterCount, randomSeed) {
  const random = seededRandom(randomSeed);
  const firstIndex = Math.floor(random() * vectors.length);
  const centroidIndexes = [firstIndex];
  const minDistances = vectors.map((vector) => cosineDistance(vector, vectors[firstIndex]));

  while (centroidIndexes.length < clusterCount) {
    const totalDistance = minDistances.reduce((total, distance) => total + distance, 0);
    let nextIndex = -1;

    if (totalDistance > 0) {
      nextIndex = chooseWeightedIndex(minDistances, totalDistance, random);
    } else {
      nextIndex = vectors.findIndex((_, index) => !centroidIndexes.includes(index));
    }

    if (nextIndex < 0 || centroidIndexes.includes(nextIndex)) {
      nextIndex = vectors.findIndex((_, index) => !centroidIndexes.includes(index));
    }
    centroidIndexes.push(nextIndex);

    for (let index = 0; index < vectors.length; index += 1) {
      minDistances[index] = Math.min(minDistances[index], cosineDistance(vectors[index], vectors[nextIndex]));
    }
  }

  return centroidIndexes.map((index) => vectors[index].slice());
}

function assignToNearestCentroids(vectors, centroids, assignments) {
  let changed = 0;
  for (let index = 0; index < vectors.length; index += 1) {
    const vector = vectors[index];
    let bestCluster = 0;
    let bestSimilarity = Number.NEGATIVE_INFINITY;

    for (let clusterId = 0; clusterId < centroids.length; clusterId += 1) {
      const similarity = dotProduct(vector, centroids[clusterId]);
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestCluster = clusterId;
      }
    }

    if (assignments[index] !== bestCluster) {
      assignments[index] = bestCluster;
      changed += 1;
    }
  }
  return changed;
}

function countAssignments(assignments, clusterCount) {
  const counts = new Array(clusterCount).fill(0);
  for (const clusterId of assignments) {
    counts[clusterId] += 1;
  }
  return counts;
}

function reseedEmptyClusters(vectors, centroids, assignments) {
  const counts = countAssignments(assignments, centroids.length);
  let changed = 0;

  for (let emptyCluster = 0; emptyCluster < counts.length; emptyCluster += 1) {
    if (counts[emptyCluster] > 0) {
      continue;
    }

    let farthestIndex = -1;
    let farthestDistance = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < vectors.length; index += 1) {
      const assignedCluster = assignments[index];
      if (counts[assignedCluster] <= 1) {
        continue;
      }

      const distance = cosineDistance(vectors[index], centroids[assignedCluster]);
      if (distance > farthestDistance || (distance === farthestDistance && index < farthestIndex)) {
        farthestDistance = distance;
        farthestIndex = index;
      }
    }

    if (farthestIndex < 0) {
      throw new Error(`Unable to re-seed empty semantic cluster ${emptyCluster}.`);
    }

    counts[assignments[farthestIndex]] -= 1;
    assignments[farthestIndex] = emptyCluster;
    counts[emptyCluster] = 1;
    changed += 1;
  }

  return changed;
}

function updateSphericalCentroids(vectors, assignments, clusterCount) {
  const dimensions = vectors[0].length;
  const sums = Array.from({ length: clusterCount }, () => new Array(dimensions).fill(0));
  const counts = new Array(clusterCount).fill(0);

  for (let index = 0; index < vectors.length; index += 1) {
    const clusterId = assignments[index];
    const vector = vectors[index];
    counts[clusterId] += 1;
    for (let dimension = 0; dimension < dimensions; dimension += 1) {
      sums[clusterId][dimension] += vector[dimension];
    }
  }

  return sums.map((sum, clusterId) => {
    if (counts[clusterId] === 0) {
      throw new Error(`Semantic cluster ${clusterId} has no members after re-seeding.`);
    }
    return normalizeVector(sum);
  });
}

function sphericalKMeans(vectors, clusterCount, maxIterations, randomSeed) {
  let centroids = initializeKMeansPlusPlus(vectors, clusterCount, randomSeed);
  const assignments = new Array(vectors.length).fill(-1);
  let iterations = 0;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const assignedChanges = assignToNearestCentroids(vectors, centroids, assignments);
    const reseededChanges = reseedEmptyClusters(vectors, centroids, assignments);
    centroids = updateSphericalCentroids(vectors, assignments, clusterCount);
    iterations = iteration + 1;

    if (assignedChanges + reseededChanges === 0) {
      break;
    }
  }

  return { assignments, iterations };
}

function groupIndexes(assignments, clusterCount) {
  const groups = Array.from({ length: clusterCount }, () => []);
  for (let index = 0; index < assignments.length; index += 1) {
    groups[assignments[index]].push(index);
  }
  return groups;
}

function buildClusterCenters(points, memberIndexes) {
  return memberIndexes.map((members, clusterId) => {
    if (members.length === 0) {
      throw new Error(`Cluster ${clusterId} has no members.`);
    }
    return centerOfMembers(points, members);
  });
}

function globalCenter(points) {
  const center = new Array(points[0].length).fill(0);
  for (const point of points) {
    for (let dimension = 0; dimension < point.length; dimension += 1) {
      center[dimension] += point[dimension];
    }
  }
  return center.map((value) => value / points.length);
}

function clampVisualCoordinate(value) {
  return Math.round(Math.min(1000, Math.max(0, value)));
}

function compressedClusterDeltas(point, center) {
  const deltas = point.map((value, dimension) => value - center[dimension]);
  const distance = Math.sqrt(deltas.reduce((total, value) => total + value * value, 0));
  if (distance === 0) {
    return deltas;
  }

  const tailDistance = Math.max(0, distance - VISUAL_OUTLIER_SOFT_RADIUS);
  const compressedDistance =
    Math.min(distance, VISUAL_OUTLIER_SOFT_RADIUS) +
    Math.log1p(tailDistance / VISUAL_OUTLIER_SOFT_RADIUS) *
      VISUAL_OUTLIER_SOFT_RADIUS *
      VISUAL_OUTLIER_TAIL_SPREAD;
  const scale = (compressedDistance / distance) * VISUAL_INTRA_CLUSTER_SPREAD;
  return deltas.map((value) => value * scale);
}

function applyVisualCompression(points, assignments, memberIndexes) {
  const centers = buildClusterCenters(points, memberIndexes);
  const atlasCenter = globalCenter(points);
  const compressedCenters = centers.map((center) =>
    center.map(
      (value, dimension) => atlasCenter[dimension] + (value - atlasCenter[dimension]) * VISUAL_CLUSTER_CENTER_SPREAD,
    ),
  );

  return points.map((point, index) => {
    const clusterId = assignments[index];
    const center = centers[clusterId];
    const compressedCenter = compressedCenters[clusterId];
    const deltas = compressedClusterDeltas(point, center);
    return point.map((value, dimension) =>
      clampVisualCoordinate(compressedCenter[dimension] + deltas[dimension]),
    );
  });
}

function toHexChannel(value) {
  return Math.round(value).toString(16).padStart(2, '0');
}

function hueToRgb(p, q, t) {
  if (t < 0) {
    t += 1;
  }
  if (t > 1) {
    t -= 1;
  }
  if (t < 1 / 6) {
    return p + (q - p) * 6 * t;
  }
  if (t < 1 / 2) {
    return q;
  }
  if (t < 2 / 3) {
    return p + (q - p) * (2 / 3 - t) * 6;
  }
  return p;
}

function hslToHex(hue, saturation, lightness) {
  const h = (((hue % 360) + 360) % 360) / 360;
  const s = saturation / 100;
  const l = lightness / 100;
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const red = hueToRgb(p, q, h + 1 / 3) * 255;
  const green = hueToRgb(p, q, h) * 255;
  const blue = hueToRgb(p, q, h - 1 / 3) * 255;
  return `#${toHexChannel(red)}${toHexChannel(green)}${toHexChannel(blue)}`;
}

function buildColorCandidates(count) {
  return Array.from({ length: Math.max(72, count * 2) }, (_, index) => ({
    hue: (index * 137.508 + 18) % 360,
    saturation: 68 + (index % 3) * 7,
    lightness: 47 + ((index * 5) % 17),
  }));
}

function colorDistance(left, right) {
  const hueDelta = Math.abs(left.hue - right.hue);
  const hueDistance = Math.min(hueDelta, 360 - hueDelta) / 180;
  const saturationDistance = Math.abs(left.saturation - right.saturation) / 100;
  const lightnessDistance = Math.abs(left.lightness - right.lightness) / 100;
  return hueDistance * 8 + saturationDistance + lightnessDistance;
}

function hueSeparation(left, right) {
  const hueDelta = Math.abs(left.hue - right.hue);
  return Math.min(hueDelta, 360 - hueDelta);
}

function assignIslandColors(centers) {
  const candidates = buildColorCandidates(centers.length);
  const assigned = new Array(centers.length);
  const remainingIndexes = new Set(candidates.map((_, index) => index));
  const nearestIds = centers.map((center, index) =>
    centers
      .map((otherCenter, otherIndex) => ({
        index: otherIndex,
        distance: otherIndex === index ? Number.POSITIVE_INFINITY : squaredDistance(center, otherCenter),
      }))
      .sort((left, right) => left.distance - right.distance || left.index - right.index)
      .slice(0, 8)
      .map((item) => item.index),
  );
  const order = centers
    .map((_, index) => {
      const nearestDistances = nearestIds[index].map((otherIndex) => squaredDistance(centers[index], centers[otherIndex]));
      return {
        index,
        crowding: nearestDistances.reduce((total, distance) => total + Math.sqrt(distance), 0),
      };
    })
    .sort((left, right) => left.crowding - right.crowding || left.index - right.index)
    .map((item) => item.index);

  for (const clusterId of order) {
    let bestCandidateIndex = [...remainingIndexes][0];
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const candidateIndex of remainingIndexes) {
      const candidate = candidates[candidateIndex];
      let weightedContrast = 0;
      let nearestContrast = Number.POSITIVE_INFINITY;
      let nearestHueSeparation = Number.POSITIVE_INFINITY;
      let globalContrast = Number.POSITIVE_INFINITY;

      for (let otherId = 0; otherId < assigned.length; otherId += 1) {
        const otherColor = assigned[otherId];
        if (!otherColor) {
          continue;
        }

        const visualDistance = Math.sqrt(squaredDistance(centers[clusterId], centers[otherId]));
        const nearbyWeight = 1 / (1 + visualDistance / 180);
        const contrast = colorDistance(candidate, otherColor);
        weightedContrast += contrast * nearbyWeight;
        globalContrast = Math.min(globalContrast, contrast);
        if (nearestIds[clusterId].includes(otherId) || nearestIds[otherId].includes(clusterId)) {
          nearestContrast = Math.min(nearestContrast, contrast);
          nearestHueSeparation = Math.min(nearestHueSeparation, hueSeparation(candidate, otherColor));
        }
      }

      const score =
        weightedContrast +
        (Number.isFinite(nearestContrast) ? nearestContrast * 100 : 0) +
        (Number.isFinite(nearestHueSeparation) ? nearestHueSeparation * 10 : 0) +
        (Number.isFinite(globalContrast) ? globalContrast * 2 : 0);

      if (score > bestScore) {
        bestScore = score;
        bestCandidateIndex = candidateIndex;
      }
    }

    assigned[clusterId] = candidates[bestCandidateIndex];
    remainingIndexes.delete(bestCandidateIndex);
  }

  return assigned.map((color) => ({
    ...color,
    hex: hslToHex(color.hue, color.saturation, color.lightness),
  }));
}

function computeInitialView(scaled3d) {
  const count = scaled3d.length;
  const centerX = scaled3d.reduce((sum, p) => sum + p[0], 0) / count;
  const centerY = scaled3d.reduce((sum, p) => sum + p[1], 0) / count;
  const centerZ = scaled3d.reduce((sum, p) => sum + p[2], 0) / count;
  const radius = Math.sqrt(
    scaled3d.reduce((sum, p) => {
      const dx = p[0] - centerX;
      const dy = p[1] - centerY;
      const dz = p[2] - centerZ;
      return sum + dx * dx + dy * dy + dz * dz;
    }, 0) / count,
  ) || 1;

  const normalized = scaled3d.map((p) => [
    (p[0] - centerX) / (radius * 2),
    (centerY - p[1]) / (radius * 2),
    (p[2] - centerZ) / (radius * 2),
  ]);

  const CULL_THRESHOLD = -1.1;
  const SAMPLES_X = 36;
  const SAMPLES_Y = 72;
  const TILT_MIN = -Math.PI * 0.35;
  const TILT_MAX = Math.PI * 0.05;
  const GRID_SIZE = 48;

  let maxVisible = 0;
  const candidates = [];

  for (let ix = 0; ix < SAMPLES_X; ix += 1) {
    const rx = TILT_MIN + (ix / (SAMPLES_X - 1)) * (TILT_MAX - TILT_MIN);
    const cosX = Math.cos(rx);
    const sinX = Math.sin(rx);

    for (let iy = 0; iy < SAMPLES_Y; iy += 1) {
      const ry = (iy / SAMPLES_Y) * Math.PI * 2;
      const cosY = Math.cos(ry);
      const sinY = Math.sin(ry);

      let visible = 0;
      const projected = [];
      for (const p of normalized) {
        const x1 = p[0] * cosY + p[2] * sinY;
        const z1 = -p[0] * sinY + p[2] * cosY;
        const y1 = p[1] * cosX - z1 * sinX;
        const z2 = p[1] * sinX + z1 * cosX;
        if (z2 >= CULL_THRESHOLD) {
          visible += 1;
          projected.push(x1, y1);
        }
      }

      if (visible > maxVisible) {
        maxVisible = visible;
      }

      let occupiedCells = 0;
      if (visible > 0) {
        const cells = new Set();
        for (let j = 0; j < projected.length; j += 2) {
          const cx = Math.floor((projected[j] + 1.5) * GRID_SIZE);
          const cy = Math.floor((projected[j + 1] + 1.5) * GRID_SIZE);
          cells.add(cx * 1000 + cy);
        }
        occupiedCells = cells.size;
      }

      candidates.push({ rx, ry, visible, occupiedCells });
    }
  }

  const visibilityThreshold = maxVisible * 0.90;
  let bestRotateX = 0;
  let bestRotateY = 0;
  let bestScore = -Infinity;

  for (const c of candidates) {
    if (c.visible < visibilityThreshold) {
      continue;
    }
    if (c.occupiedCells > bestScore) {
      bestScore = c.occupiedCells;
      bestRotateX = c.rx;
      bestRotateY = c.ry;
    }
  }

  return {
    rotateX: Math.round(bestRotateX * 1000) / 1000,
    rotateY: Math.round(bestRotateY * 1000) / 1000,
    zoom: VISUAL_INITIAL_ZOOM,
    offsetX: 0,
    offsetY: 0,
  };
}

function pointColorFromIsland(coordinate, islandColor) {
  const x = coordinate[0] / 1000;
  const y = coordinate[1] / 1000;
  const z = coordinate[2] / 1000;
  const hue = islandColor.hue + (z - 0.5) * 10;
  const saturation = islandColor.saturation + (x - 0.5) * 10;
  const lightness = islandColor.lightness + (y - 0.5) * 12;
  return hslToHex(hue, saturation, lightness);
}

function rawTokens(text) {
  return text
    .normalize('NFC')
    .toLowerCase()
    .replace(/[’‘`]/g, "'")
    .match(/[\p{L}\p{N}']+/gu)
    ?.map((token) => {
      let normalized = token.replace(/^'+|'+$/g, '');
      for (const suffix of ["'s", "'re", "'ve", "'ll", "'m", "'d", "n't"]) {
        if (normalized.endsWith(suffix) && normalized.length > suffix.length) {
          normalized = normalized.slice(0, -suffix.length);
          break;
        }
      }
      return normalized;
    })
    .filter(Boolean) ?? [];
}

function tokenize(text) {
  return rawTokens(text)
    .filter((token) => token.length > 2 && !token.includes("'") && !STOPWORDS.has(token) && !/^\p{N}+$/u.test(token));
}

function phrasesForText(text) {
  const tokens = tokenize(text);
  const phrases = [...tokens];
  for (let index = 0; index < tokens.length - 1; index += 1) {
    const phrase = `${tokens[index]} ${tokens[index + 1]}`;
    phrases.push(phrase);
  }
  return phrases;
}

function buildPhraseStats(entries, assignments, clusterCount) {
  const clusterCounts = Array.from({ length: clusterCount }, () => new Map());
  const clusterTotals = new Array(clusterCount).fill(0);

  for (let index = 0; index < entries.length; index += 1) {
    const phrases = phrasesForText(entries[index].en);
    for (const phrase of phrases) {
      const counts = clusterCounts[assignments[index]];
      counts.set(phrase, (counts.get(phrase) ?? 0) + 1);
      clusterTotals[assignments[index]] += 1;
    }
  }

  const phraseClusterCounts = new Map();
  for (const counts of clusterCounts) {
    for (const phrase of counts.keys()) {
      phraseClusterCounts.set(phrase, (phraseClusterCounts.get(phrase) ?? 0) + 1);
    }
  }

  return clusterCounts.map((counts, clusterId) =>
    [...counts.entries()]
      .map(([phrase, count]) => ({
        phrase,
        count,
        score:
          (count / Math.max(1, clusterTotals[clusterId])) *
          Math.log(1 + clusterCount / (phraseClusterCounts.get(phrase) ?? 1)),
      }))
      .sort((left, right) => right.score - left.score || right.count - left.count || left.phrase.localeCompare(right.phrase)),
  );
}

function titleCasePhrase(phrase) {
  return phrase
    .split(' ')
    .map((word) => {
      if (word === 'abs' || word === 'gt3') {
        return word.toUpperCase();
      }
      return `${word.slice(0, 1).toUpperCase()}${word.slice(1)}`;
    })
    .join(' ');
}

function selectDistinctPhrases(phrases, limit) {
  const selected = [];
  for (const phrase of phrases) {
    if (selected.some((item) => item.includes(phrase) || phrase.includes(item))) {
      continue;
    }
    selected.push(phrase);
    if (selected.length === limit) {
      break;
    }
  }
  return selected;
}

function phraseTerms(phrase) {
  return phrase.split(' ');
}

function isWeakLabelPhrase(phrase) {
  const terms = phraseTerms(phrase);
  return (
    terms.every((term) => LABEL_WEAK_TERMS.has(term)) ||
    terms.some((term) => term === 'bcy' || term === 'com' || term === 'nur') ||
    /\b(\w+)\b \1\b/.test(phrase)
  );
}

function conceptsForText(text) {
  const tokens = new Set(rawTokens(text));
  return LABEL_CONCEPTS
    .filter((concept) => [...concept.terms].some((term) => tokens.has(term)))
    .map((concept) => concept.label);
}

function labelCandidatesForText(text) {
  return [...phrasesForText(text).filter((phrase) => !isWeakLabelPhrase(phrase)), ...conceptsForText(text)];
}

function buildLabelStats(entries, assignments, clusterCount) {
  const globalEntryCounts = new Map();
  const clusterEntryCounts = Array.from({ length: clusterCount }, () => new Map());
  const clusterSizes = new Array(clusterCount).fill(0);

  for (let index = 0; index < entries.length; index += 1) {
    const clusterId = assignments[index];
    clusterSizes[clusterId] += 1;
    const seen = new Set(labelCandidatesForText(entries[index].en));

    for (const phrase of seen) {
      globalEntryCounts.set(phrase, (globalEntryCounts.get(phrase) ?? 0) + 1);
      const counts = clusterEntryCounts[clusterId];
      counts.set(phrase, (counts.get(phrase) ?? 0) + 1);
    }
  }

  return clusterEntryCounts.map((counts, clusterId) => {
    const clusterSize = clusterSizes[clusterId];
    const outsideSize = entries.length - clusterSize;
    const minEntryCount = Math.max(4, Math.ceil(clusterSize * 0.004));

    return [...counts.entries()]
      .map(([phrase, count]) => {
        const globalCount = globalEntryCounts.get(phrase) ?? count;
        const outsideCount = globalCount - count;
        const coverage = count / clusterSize;
        const outsideCoverage = outsideCount / outsideSize;
        const lift = Math.log2((coverage + 0.001) / (outsideCoverage + 0.001));
        const exclusivity = count / globalCount;
        const isConcept = LABEL_CONCEPT_LABELS.has(phrase);
        const score =
          Math.log1p(count) *
          Math.max(0, lift) *
          Math.pow(coverage, 0.62) *
          (0.4 + 0.6 * exclusivity) *
          (isConcept ? 0.85 : 1);
        return { phrase, count, score, lift };
      })
      .filter((item) => item.count >= minEntryCount && item.lift > 0.45)
      .sort((left, right) => right.score - left.score || right.count - left.count || left.phrase.localeCompare(right.phrase));
  });
}

function canonicalLabelRoot(phrase) {
  return phrase.replace(/s$/, '');
}

function selectDistinctLabelStats(stats, limit) {
  const selected = [];
  for (const item of stats) {
    const terms = new Set(phraseTerms(item.phrase).map(canonicalLabelRoot));
    const overlapsExisting = selected.some((selectedItem) => {
      const selectedTerms = new Set(phraseTerms(selectedItem.phrase).map(canonicalLabelRoot));
      let overlap = 0;
      for (const term of terms) {
        if (selectedTerms.has(term)) {
          overlap += 1;
        }
      }
      return overlap >= Math.min(terms.size, selectedTerms.size);
    });

    if (!overlapsExisting) {
      selected.push(item);
      if (selected.length === limit) {
        break;
      }
    }
  }
  return selected;
}

function labelFromStats(stats, fallbackPhrases) {
  const selected = selectDistinctLabelStats(stats, 2);
  if (selected.length > 0) {
    return selected.map((item) => titleCasePhrase(item.phrase)).join(' / ');
  }
  const fallback = selectDistinctPhrases(fallbackPhrases.filter((phrase) => !isWeakLabelPhrase(phrase)), 2);
  return fallback.length ? fallback.map(titleCasePhrase).join(' / ') : 'Mixed Lines';
}

function labelUsesConcept(label) {
  return label
    .split('/')
    .map((phrase) => phrase.trim().toLowerCase())
    .some((phrase) => LABEL_CONCEPT_LABELS.has(phrase));
}

function descriptionFromPhrases(phrases, label, size, videoCount) {
  const strongPhrases = phrases.filter((phrase) => !isWeakLabelPhrase(phrase));
  const labelPhrases = label
    .split('/')
    .map((phrase) => phrase.trim())
    .filter(Boolean);
  const phraseWords = strongPhrases.length >= 2 && !labelUsesConcept(label)
    ? selectDistinctPhrases(strongPhrases, 5).map(titleCasePhrase)
    : [];
  const themeWords = phraseWords.length >= 2 ? phraseWords : labelPhrases;
  const scope = `${size.toLocaleString('en-US')} lines from ${videoCount.toLocaleString('en-US')} videos`;
  return themeWords.length ? `${themeWords.join(', ')}. ${scope}.` : scope;
}

// Compactness describes the dense core of an island, so distant fragments do not define the score.
function compactRadius(members, scaled3d, center) {
  const distances = members
    .map((index) => Math.sqrt(squaredDistance(scaled3d[index], center)))
    .sort((left, right) => left - right);
  const coreCount = Math.max(1, Math.ceil(distances.length * COMPACTNESS_CORE_SHARE));
  const coreDistance = distances
    .slice(0, coreCount)
    .reduce((total, distance) => total + distance, 0);
  return coreDistance / coreCount;
}

function buildClusters(entries, assignments, scaled3d, centers, islandColors, memberIndexes) {
  const clusterCount = centers.length;
  const phraseStats = buildPhraseStats(entries, assignments, clusterCount);
  const labelStats = buildLabelStats(entries, assignments, clusterCount);

  return memberIndexes.map((members, clusterId) => {
    const topPhrases = phraseStats[clusterId].slice(0, PHRASE_COUNT).map((item) => item.phrase);
    const label = labelFromStats(labelStats[clusterId], topPhrases);
    const videoIds = new Set(members.map((index) => entries[index].videoId));
    const center3d = [
      Math.round(members.reduce((total, index) => total + scaled3d[index][0], 0) / members.length),
      Math.round(members.reduce((total, index) => total + scaled3d[index][1], 0) / members.length),
      Math.round(members.reduce((total, index) => total + scaled3d[index][2], 0) / members.length),
    ];
    const rawCenter = centers[clusterId];
    const medoid = members
      .map((index) => ({
        index,
        distance: squaredDistance(scaled3d[index], rawCenter),
      }))
      .sort(
        (left, right) =>
          left.distance - right.distance ||
          entries[left.index].entryId.localeCompare(entries[right.index].entryId),
      )[0];

    return {
      id: clusterId,
      label,
      description: descriptionFromPhrases(topPhrases, label, members.length, videoIds.size),
      color: islandColors[clusterId].hex,
      size: members.length,
      videoCount: videoIds.size,
      x3d: center3d[0],
      y3d: center3d[1],
      z3d: center3d[2],
      topPhrases,
      medoidEntryId: entries[medoid.index].entryId,
      rawCompactRadius: compactRadius(members, scaled3d, rawCenter),
    };
  });
}

function readSemanticRows(SQL) {
  const db = new SQL.Database(fs.readFileSync(DB_PATH));
  const statement = db.prepare(`
    SELECT m.video_id, m.seg_index, m.en, m.zh, m.start_ms, m.end_ms, v.vector
    FROM tm_main AS m
    JOIN tm_vectors AS v USING (content_sha)
    WHERE v.model_id = $modelId
    ORDER BY m.video_id, m.seg_index
  `);
  statement.bind({ $modelId: MODEL_ID });

  const entries = [];
  const vectors = [];
  let vectorDim = 0;

  while (statement.step()) {
    const row = statement.getAsObject();
    const blob = row.vector;
    const dim = blob.byteLength / 4;
    vectorDim = vectorDim || dim;
    const typed = new Float32Array(blob.buffer.slice(blob.byteOffset, blob.byteOffset + blob.byteLength));
    const vector = normalizeVector(Array.from(typed));
    vectors.push(vector);
    entries.push({
      entryId: `${row.video_id}#${row.seg_index}`,
      videoId: String(row.video_id),
      segIndex: Number(row.seg_index),
      en: String(row.en),
      zh: String(row.zh),
      startMs: row.start_ms === null ? null : Number(row.start_ms),
      endMs: row.end_ms === null ? null : Number(row.end_ms),
    });
  }

  statement.free();
  db.close();

  return { entries, vectors, vectorDim };
}

async function main() {
  const SQL = await initSqlJs({ locateFile: () => SQL_WASM_PATH });

  const { entries, vectors, vectorDim } = readSemanticRows(SQL);
  console.log(`Read ${entries.length.toLocaleString()} MiniLM vectors (${vectorDim} dimensions).`);

  console.log(`Running spherical k-means (${SEMANTIC_CLUSTER_COUNT} semantic islands).`);
  const { assignments, iterations } = sphericalKMeans(
    vectors,
    SEMANTIC_CLUSTER_COUNT,
    SEMANTIC_KMEANS_MAX_ITERATIONS,
    SEMANTIC_KMEANS_SEED,
  );
  console.log(`  converged in ${iterations} iteration${iterations === 1 ? '' : 's'}.`);

  console.log('Running supervised UMAP 3D.');
  const clusterCount = SEMANTIC_CLUSTER_COUNT;
  const memberIndexes = groupIndexes(assignments, clusterCount);
  const scaled3d = applyVisualCompression(scaleCoordinates(runUmap(vectors, assignments)), assignments, memberIndexes);
  const centers = buildClusterCenters(scaled3d, memberIndexes);
  const islandColors = assignIslandColors(centers);
  const clusters = buildClusters(entries, assignments, scaled3d, centers, islandColors, memberIndexes);
  const compactRadii = clusters
    .map((cluster) => cluster.rawCompactRadius)
    .sort((left, right) => left - right);
  const compactReference = percentile(compactRadii, COMPACTNESS_REFERENCE_PERCENTILE);
  for (const cluster of clusters) {
    cluster.compactness = compactReference > 0
      ? Math.max(
          1,
          Math.min(100, Math.round((compactReference / (compactReference + cluster.rawCompactRadius)) * 100)),
        )
      : 100;
    delete cluster.rawCompactRadius;
  }
  console.log('Computing optimal initial view.');
  const initialView = computeInitialView(scaled3d);
  console.log(`  rotateX=${initialView.rotateX}, rotateY=${initialView.rotateY}`);

  const points = entries.map((entry, index) => ({
    entryId: entry.entryId,
    videoId: entry.videoId,
    segIndex: entry.segIndex,
    en: entry.en,
    zh: entry.zh,
    startMs: entry.startMs,
    endMs: entry.endMs,
    x3d: scaled3d[index][0],
    y3d: scaled3d[index][1],
    z3d: scaled3d[index][2],
    clusterId: assignments[index],
    color: pointColorFromIsland(scaled3d[index], islandColors[assignments[index]]),
  }));

  const payload = {
    version: 9,
    projection: 'umap-3d',
    clusterAlgorithm: 'minilm-kmeans-supervised-umap-islands',
    clusterBasis: 'minilm-spherical-kmeans',
    generatedAt: new Date().toISOString(),
    sourceDb: path.basename(DB_PATH),
    modelId: MODEL_ID,
    pointCount: points.length,
    vectorDim,
    clusterCount,
    semanticKMeans: {
      clusters: SEMANTIC_CLUSTER_COUNT,
      iterations: SEMANTIC_KMEANS_MAX_ITERATIONS,
      randomSeed: SEMANTIC_KMEANS_SEED,
      distance: 'cosine',
    },
    umap: {
      neighbors: UMAP_NEIGHBORS,
      minDist: UMAP_MIN_DIST,
      spread: UMAP_SPREAD,
      epochs: UMAP_EPOCHS,
      randomSeed: RANDOM_SEED,
      supervisedTargetWeight: UMAP_SUPERVISED_TARGET_WEIGHT,
    },
    visualPostProcessing: {
      clusterCenterSpread: VISUAL_CLUSTER_CENTER_SPREAD,
      intraClusterSpread: VISUAL_INTRA_CLUSTER_SPREAD,
      outlierSoftRadius: VISUAL_OUTLIER_SOFT_RADIUS,
      outlierTailSpread: VISUAL_OUTLIER_TAIL_SPREAD,
      initialZoom: VISUAL_INITIAL_ZOOM,
    },
    initialView,
    clusters,
    points,
  };

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(payload), 'utf8');
  console.log(`Wrote ${OUTPUT_PATH}.`);
}

main();
