import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import initSqlJs from 'sql.js';
import { UMAP } from 'umap-js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const DB_PATH = path.join(REPO_ROOT, 'public', 'data', 'tm_misha_minilm.db');
const OUTPUT_PATH = path.join(REPO_ROOT, 'public', 'data', 'startup-visualizations.json');
const SQL_WASM_PATH = path.join(REPO_ROOT, 'node_modules', 'sql.js', 'dist', 'sql-wasm.wasm');

const MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2';
const RANDOM_SEED = 24_031;
const UMAP_NEIGHBORS = 22;
const UMAP_MIN_DIST = 0.035;
const UMAP_SPREAD = 1.2;
const UMAP_EPOCHS = 220;
const CLUSTER_ITERATIONS = 16;
const SAMPLE_COUNT = 4;
const PHRASE_COUNT = 8;
const PALETTE = [
  '#2fb7a5',
  '#f06449',
  '#8ab17d',
  '#f2b84b',
  '#7c6fd3',
  '#4d96c8',
  '#d668a5',
  '#58a36d',
  '#d98c45',
  '#6fc2d0',
  '#b28bd8',
  '#c4a746',
  '#e2766f',
  '#5f9ea0',
  '#a6a15e',
  '#88b7e0',
];
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

function chooseClusterCount(entryCount) {
  return Math.max(14, Math.min(24, Math.round(Math.sqrt(entryCount / 48))));
}

function squaredDistance(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    const delta = left[index] - right[index];
    total += delta * delta;
  }
  return total;
}

function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    total += left[index] * right[index];
  }
  return total;
}

function initializeSemanticCenters(vectors, clusterCount) {
  const random = seededRandom(RANDOM_SEED + 4_000);
  const firstIndex = Math.floor(random() * vectors.length);
  const centers = [vectors[firstIndex].slice()];
  const nearestDistances = vectors.map((vector) => Math.max(0, 1 - dot(vector, centers[0])));

  while (centers.length < clusterCount) {
    const totalDistance = nearestDistances.reduce((total, value) => total + value, 0);
    let threshold = random() * totalDistance;
    let nextIndex = vectors.length - 1;

    for (let index = 0; index < nearestDistances.length; index += 1) {
      threshold -= nearestDistances[index];
      if (threshold <= 0) {
        nextIndex = index;
        break;
      }
    }

    const center = vectors[nextIndex].slice();
    centers.push(center);

    for (let index = 0; index < vectors.length; index += 1) {
      const distance = Math.max(0, 1 - dot(vectors[index], center));
      if (distance < nearestDistances[index]) {
        nearestDistances[index] = distance;
      }
    }
  }

  return centers;
}

function clusterSemanticVectors(vectors, scaled2d, clusterCount) {
  let centers = initializeSemanticCenters(vectors, clusterCount);
  let assignments = new Array(vectors.length).fill(0);

  for (let iteration = 0; iteration < CLUSTER_ITERATIONS; iteration += 1) {
    const sums = Array.from({ length: clusterCount }, () => new Array(vectors[0].length).fill(0));
    const counts = new Array(clusterCount).fill(0);
    const bestScores = new Array(vectors.length).fill(-Infinity);

    for (let vectorIndex = 0; vectorIndex < vectors.length; vectorIndex += 1) {
      const vector = vectors[vectorIndex];
      let bestCluster = 0;
      let bestScore = -Infinity;

      for (let clusterId = 0; clusterId < clusterCount; clusterId += 1) {
        const score = dot(vector, centers[clusterId]);
        if (score > bestScore) {
          bestScore = score;
          bestCluster = clusterId;
        }
      }

      assignments[vectorIndex] = bestCluster;
      bestScores[vectorIndex] = bestScore;
      counts[bestCluster] += 1;
      for (let dimension = 0; dimension < vector.length; dimension += 1) {
        sums[bestCluster][dimension] += vector[dimension];
      }
    }

    centers = centers.map((center, clusterId) => {
      if (counts[clusterId] === 0) {
        const replacementIndex = bestScores.indexOf(Math.min(...bestScores));
        return vectors[replacementIndex].slice();
      }
      return normalizeVector(sums[clusterId]);
    });
  }

  const clusterPositions = centers.map((center, id) => {
    const members = assignments
      .map((clusterId, index) => (clusterId === id ? index : -1))
      .filter((index) => index >= 0);
    return {
      id,
      x: members.reduce((total, index) => total + scaled2d[index][0], 0) / members.length,
      y: members.reduce((total, index) => total + scaled2d[index][1], 0) / members.length,
    };
  });
  const orderedClusterIds = clusterPositions
    .sort((left, right) => left.x - right.x || left.y - right.y)
    .map((cluster) => cluster.id);
  const remap = new Map(orderedClusterIds.map((clusterId, index) => [clusterId, index]));

  assignments = assignments.map((clusterId) => remap.get(clusterId));
  centers = orderedClusterIds.map((clusterId) => centers[clusterId]);

  return { assignments, centers };
}

function tokenize(text) {
  return text
    .toLowerCase()
    .match(/[a-z0-9']+/g)
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
    .filter((token) => token.length > 2 && !STOPWORDS.has(token) && !/^\d+$/.test(token)) ?? [];
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

function labelFromPhrases(phrases) {
  const selected = [];
  for (const phrase of phrases) {
    if (selected.some((item) => item.includes(phrase) || phrase.includes(item))) {
      continue;
    }
    selected.push(phrase);
    if (selected.length === 2) {
      break;
    }
  }

  return selected.map(titleCasePhrase).join(' / ');
}

function buildClusters(entries, vectors, assignments, scaled2d, scaled3d, centers) {
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
    const sampleIndexes = members
      .map((index) => ({
        index,
        score: dot(vectors[index], rawCenter),
      }))
      .sort((left, right) => right.score - left.score || entries[left.index].entryId.localeCompare(entries[right.index].entryId))
      .slice(0, SAMPLE_COUNT)
      .map((item) => item.index);

    return {
      id: clusterId,
      label: labelFromPhrases(phraseStats[clusterId]),
      labelMode: 'descriptive',
      labelConfidence: 1,
      color: PALETTE[clusterId % PALETTE.length],
      size: members.length,
      videoCount: videoIds.size,
      x: center2d[0],
      y: center2d[1],
      z: center3d[2],
      x3d: center3d[0],
      y3d: center3d[1],
      z3d: center3d[2],
      topPhrases: phraseStats[clusterId],
      medoidEntryId: entries[sampleIndexes[0]].entryId,
      representativeEntryIds: sampleIndexes.map((index) => entries[index].entryId),
      samples: sampleIndexes.map((index) => {
        const entry = entries[index];
        return {
          entryId: entry.entryId,
          videoId: entry.videoId,
          segIndex: entry.segIndex,
          en: entry.en,
          zh: entry.zh,
          blockName: entry.blockName,
        };
      }),
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
  const clusterCount = chooseClusterCount(entries.length);
  const { assignments, centers } = clusterSemanticVectors(vectors, scaled2d, clusterCount);
  const clusters = buildClusters(entries, vectors, assignments, scaled2d, scaled3d, centers);
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
  }));

  const payload = {
    version: 6,
    projection: 'umap-2d-3d',
    clusterAlgorithm: 'semantic-spherical-kmeans',
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
    clusters,
    points,
  };

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(payload), 'utf8');
  console.log(`Wrote ${OUTPUT_PATH}.`);
}

main();
