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
const ISLAND_NEIGHBOR_COUNT = 7;
const ISLAND_EDGE_STRENGTH = 0.7;
const ISLAND_GRID_CELL_SIZE = 44;
const ISLAND_CANDIDATE_COUNT = 72;
const ISLAND_MIN_SEED_SIZE = 40;
const PHRASE_COUNT = 8;
const STOPWORDS = new Set(
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
  `.trim().split(/\s+/),
);

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

function runUmap(vectors, nComponents, seedOffset) {
  const umap = new UMAP({
    nComponents,
    nNeighbors: UMAP_NEIGHBORS,
    minDist: UMAP_MIN_DIST,
    spread: UMAP_SPREAD,
    nEpochs: UMAP_EPOCHS,
    distanceFn: cosineDistance,
    random: seededRandom(RANDOM_SEED + seedOffset),
  });
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

function cellKey(x, y, z) {
  return `${x},${y},${z}`;
}

function buildCoordinateGrid(points) {
  const buckets = new Map();
  const pointCells = [];

  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const cell = point.map((value) => Math.floor(value / ISLAND_GRID_CELL_SIZE));
    const key = cellKey(cell[0], cell[1], cell[2]);
    const bucket = buckets.get(key) ?? [];
    bucket.push(index);
    buckets.set(key, bucket);
    pointCells.push(cell);
  }

  return { buckets, pointCells };
}

function nearbyCandidateIndexes(grid, cell) {
  const candidates = [];
  const maxShell = Math.ceil(1000 / ISLAND_GRID_CELL_SIZE) + 1;

  for (let shell = 0; candidates.length < ISLAND_CANDIDATE_COUNT && shell <= maxShell; shell += 1) {
    for (let dx = -shell; dx <= shell; dx += 1) {
      for (let dy = -shell; dy <= shell; dy += 1) {
        for (let dz = -shell; dz <= shell; dz += 1) {
          if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== shell) {
            continue;
          }

          const bucket = grid.buckets.get(cellKey(cell[0] + dx, cell[1] + dy, cell[2] + dz));
          if (bucket) {
            candidates.push(...bucket);
          }
        }
      }
    }
  }

  return candidates;
}

function buildNearestNeighbors(points) {
  const grid = buildCoordinateGrid(points);
  const neighbors = [];
  const neighborDistances = [];

  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const candidates = nearbyCandidateIndexes(grid, grid.pointCells[index]);
    const nearest = candidates
      .filter((candidateIndex) => candidateIndex !== index)
      .map((candidateIndex) => ({
        index: candidateIndex,
        distance: squaredDistance(point, points[candidateIndex]),
      }))
      .sort((left, right) => left.distance - right.distance || left.index - right.index)
      .slice(0, ISLAND_NEIGHBOR_COUNT);

    neighbors.push(nearest.map((candidate) => candidate.index));
    neighborDistances.push(nearest.at(-1)?.distance ?? Number.POSITIVE_INFINITY);
  }

  return { neighbors, neighborDistances };
}

function createDisjointSet(size) {
  const parents = Array.from({ length: size }, (_, index) => index);

  function find(index) {
    while (parents[index] !== index) {
      parents[index] = parents[parents[index]];
      index = parents[index];
    }
    return index;
  }

  function union(left, right) {
    const leftRoot = find(left);
    const rightRoot = find(right);
    if (leftRoot !== rightRoot) {
      parents[rightRoot] = leftRoot;
    }
  }

  return { find, union };
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

function componentSort(left, right) {
  return (
    right.members.length - left.members.length ||
    left.center[0] - right.center[0] ||
    left.center[1] - right.center[1] ||
    left.center[2] - right.center[2]
  );
}

function mergeComponentIntoIsland(island, component) {
  const islandSize = island.members.length;
  const componentSize = component.members.length;
  const nextSize = islandSize + componentSize;

  island.center = island.center.map(
    (value, dimension) => (value * islandSize + component.center[dimension] * componentSize) / nextSize,
  );
  island.members.push(...component.members);
}

function clusterMutualKnnIslands(points) {
  const { neighbors, neighborDistances } = buildNearestNeighbors(points);
  const neighborSets = neighbors.map((indexes) => new Set(indexes));
  const disjointSet = createDisjointSet(points.length);

  for (let index = 0; index < points.length; index += 1) {
    for (const neighborIndex of neighbors[index]) {
      if (neighborIndex <= index || !neighborSets[neighborIndex].has(index)) {
        continue;
      }

      const distance = squaredDistance(points[index], points[neighborIndex]);
      const localLimit = Math.min(neighborDistances[index], neighborDistances[neighborIndex]) * ISLAND_EDGE_STRENGTH;
      if (distance <= localLimit) {
        disjointSet.union(index, neighborIndex);
      }
    }
  }

  const componentMap = new Map();
  for (let index = 0; index < points.length; index += 1) {
    const root = disjointSet.find(index);
    const members = componentMap.get(root) ?? [];
    members.push(index);
    componentMap.set(root, members);
  }

  const components = [...componentMap.values()].map((members) => ({
    members,
    center: centerOfMembers(points, members),
  }));
  const islands = components
    .filter((component) => component.members.length >= ISLAND_MIN_SEED_SIZE)
    .sort(componentSort)
    .map((component) => ({
      members: [...component.members],
      center: component.center.slice(),
    }));
  const fragments = components
    .filter((component) => component.members.length < ISLAND_MIN_SEED_SIZE)
    .sort(componentSort);

  for (const fragment of fragments) {
    let bestIsland = islands[0];
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const island of islands) {
      const distance = squaredDistance(fragment.center, island.center);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIsland = island;
      }
    }
    mergeComponentIntoIsland(bestIsland, fragment);
  }

  islands.sort((left, right) =>
    left.center[0] - right.center[0] ||
    left.center[1] - right.center[1] ||
    left.center[2] - right.center[2],
  );

  const assignments = new Array(points.length);
  for (let islandId = 0; islandId < islands.length; islandId += 1) {
    for (const member of islands[islandId].members) {
      assignments[member] = islandId;
    }
  }

  return {
    assignments,
    centers: islands.map((island) => island.center),
  };
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
    zoom: 1.18,
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

function tokenize(text) {
  return text
    .normalize('NFC')
    .toLowerCase()
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
    .filter((token) => token.length > 2 && !STOPWORDS.has(token) && !/^\p{N}+$/u.test(token)) ?? [];
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
  const globalEntryCounts = new Map();
  const clusterCounts = Array.from({ length: clusterCount }, () => new Map());

  for (let index = 0; index < entries.length; index += 1) {
    const phrases = phrasesForText(entries[index].en);
    const seen = new Set(phrases);
    for (const phrase of seen) {
      globalEntryCounts.set(phrase, (globalEntryCounts.get(phrase) ?? 0) + 1);
    }
    for (const phrase of phrases) {
      const counts = clusterCounts[assignments[index]];
      counts.set(phrase, (counts.get(phrase) ?? 0) + 1);
    }
  }

  return clusterCounts.map((counts) =>
    [...counts.entries()]
      .map(([phrase, count]) => ({
        phrase,
        score: count * Math.log(1 + entries.length / (globalEntryCounts.get(phrase) ?? 1)),
      }))
      .sort((left, right) => right.score - left.score || left.phrase.localeCompare(right.phrase))
      .slice(0, PHRASE_COUNT)
      .map((item) => item.phrase),
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

function labelFromPhrases(phrases) {
  const selected = selectDistinctPhrases(phrases, 2);
  return selected.length ? selected.map(titleCasePhrase).join(' / ') : 'Mixed Lines';
}

function descriptionFromPhrases(phrases, size, videoCount) {
  const themeWords = selectDistinctPhrases(phrases, 5).map(titleCasePhrase);
  const scope = `${size.toLocaleString('en-US')} lines from ${videoCount.toLocaleString('en-US')} videos`;
  return themeWords.length ? `${themeWords.join(', ')}. ${scope}.` : scope;
}

function buildClusters(entries, assignments, scaled2d, scaled3d, centers, islandColors) {
  const clusterCount = centers.length;
  const phraseStats = buildPhraseStats(entries, assignments, clusterCount);
  const memberIndexes = Array.from({ length: clusterCount }, () => []);

  for (let index = 0; index < assignments.length; index += 1) {
    memberIndexes[assignments[index]].push(index);
  }

  return memberIndexes.map((members, clusterId) => {
    const videoIds = new Set(members.map((index) => entries[index].videoId));
    const center2d = [
      Math.round(members.reduce((total, index) => total + scaled2d[index][0], 0) / members.length),
      Math.round(members.reduce((total, index) => total + scaled2d[index][1], 0) / members.length),
    ];
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

    const avgDist = Math.sqrt(
      members.reduce((total, index) => total + squaredDistance(scaled3d[index], rawCenter), 0) / members.length,
    );

    return {
      id: clusterId,
      label: labelFromPhrases(phraseStats[clusterId]),
      description: descriptionFromPhrases(phraseStats[clusterId], members.length, videoIds.size),
      color: islandColors[clusterId].hex,
      size: members.length,
      videoCount: videoIds.size,
      x: center2d[0],
      y: center2d[1],
      z: center3d[2],
      x3d: center3d[0],
      y3d: center3d[1],
      z3d: center3d[2],
      topPhrases: phraseStats[clusterId],
      medoidEntryId: entries[medoid.index].entryId,
      rawAvgDist: avgDist,
    };
  });
}

function readSemanticRows(SQL) {
  const db = new SQL.Database(fs.readFileSync(DB_PATH));
  const statement = db.prepare(`
    SELECT m.video_id, m.seg_index, m.en, m.zh, m.block_name, m.start_ms, m.end_ms, v.vector
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
      blockName: String(row.block_name ?? ''),
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

  console.log('Running UMAP 2D.');
  const coords2d = runUmap(vectors, 2, 0);
  console.log('Running UMAP 3D.');
  const coords3d = runUmap(vectors, 3, 1);

  const scaled2d = scaleCoordinates(coords2d);
  const scaled3d = scaleCoordinates(coords3d);
  const { assignments, centers } = clusterMutualKnnIslands(scaled3d);
  const clusterCount = centers.length;
  const islandColors = assignIslandColors(centers);
  const clusters = buildClusters(entries, assignments, scaled2d, scaled3d, centers, islandColors);
  const maxAvgDist = Math.max(...clusters.map((cluster) => cluster.rawAvgDist));
  for (const cluster of clusters) {
    cluster.compactness = maxAvgDist > 0 ? Math.round((1 - cluster.rawAvgDist / maxAvgDist) * 100) : 100;
    delete cluster.rawAvgDist;
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
    blockName: entry.blockName,
    startMs: entry.startMs,
    endMs: entry.endMs,
    x: scaled2d[index][0],
    y: scaled2d[index][1],
    z: scaled3d[index][2],
    x3d: scaled3d[index][0],
    y3d: scaled3d[index][1],
    z3d: scaled3d[index][2],
    clusterId: assignments[index],
    color: pointColorFromIsland(scaled3d[index], islandColors[assignments[index]]),
  }));

  const payload = {
    version: 8,
    projection: 'umap-2d-3d',
    clusterAlgorithm: 'umap-3d-mutual-knn-islands',
    clusterBasis: 'umap-3d-mutual-knn-island',
    generatedAt: new Date().toISOString(),
    sourceDb: path.basename(DB_PATH),
    modelId: MODEL_ID,
    pointCount: points.length,
    vectorDim,
    clusterCount,
    umap: {
      neighbors: UMAP_NEIGHBORS,
      minDist: UMAP_MIN_DIST,
      spread: UMAP_SPREAD,
      epochs: UMAP_EPOCHS,
      randomSeed: RANDOM_SEED,
    },
    mutualKnn: {
      neighbors: ISLAND_NEIGHBOR_COUNT,
      edgeStrength: ISLAND_EDGE_STRENGTH,
      gridCellSize: ISLAND_GRID_CELL_SIZE,
      candidateCount: ISLAND_CANDIDATE_COUNT,
      minSeedSize: ISLAND_MIN_SEED_SIZE,
    },
    initialView,
    clusters,
    points,
  };

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(payload), 'utf8');
  console.log(`Wrote ${OUTPUT_PATH}.`);
}

main();
