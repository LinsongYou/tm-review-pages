import json
import math
import random
import re
import sqlite3
import struct
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "tm_misha_minilm.db"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "semantic-landscape.json"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_COUNT = 10
PCA_ITERATIONS = 18
RANDOM_SEED = 7
PERCENTILE_CLIP = 0.02
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "so",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "up",
    "out",
    "off",
    "as",
    "is",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "me",
    "my",
    "our",
    "their",
    "be",
    "am",
    "are",
    "was",
    "were",
    "been",
    "being",
    "do",
    "does",
    "did",
    "done",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "just",
    "like",
    "yeah",
    "yes",
    "no",
    "not",
    "oh",
    "okay",
    "all",
    "right",
    "really",
    "very",
    "pretty",
    "much",
    "there",
    "here",
    "what",
    "when",
    "where",
    "who",
    "how",
    "why",
    "one",
    "two",
    "three",
    "got",
    "get",
    "going",
    "go",
    "come",
    "now",
    "well",
    "also",
    "because",
    "know",
    "think",
    "some",
    "more",
    "see",
    "bit",
    "lets",
    "then",
    "maybe",
    "need",
    "something",
    "still",
    "about",
    "even",
    "gonna",
    "too",
    "first",
    "say",
    "dont",
    "thats",
    "good",
    "nice",
    "car",
    "cars",
    "time",
}
PALETTE = [
    "#84a98c",
    "#f4a261",
    "#58a4b0",
    "#e76f51",
    "#8d99ae",
    "#c9ada7",
    "#e9c46a",
    "#7fb069",
    "#6d597a",
    "#4d908e",
]


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / magnitude for value in vector]


def percentile_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0

    ordered = sorted(values)
    low_index = min(len(ordered) - 1, max(0, int(len(ordered) * PERCENTILE_CLIP)))
    high_index = min(len(ordered) - 1, max(low_index, int(len(ordered) * (1 - PERCENTILE_CLIP))))
    return ordered[low_index], ordered[high_index]


def scale_value(value: float, low: float, high: float) -> int:
    if high <= low:
        return 500

    clamped = min(high, max(low, value))
    return round(((clamped - low) / (high - low)) * 1000)


def kmeans_2d(points: list[tuple[float, float]], cluster_count: int) -> tuple[list[int], list[tuple[float, float]]]:
    random.seed(RANDOM_SEED)
    centers = [list(points[index]) for index in random.sample(range(len(points)), cluster_count)]
    assignments = [0] * len(points)

    for _ in range(14):
        changed = False
        for index, (x_value, y_value) in enumerate(points):
            best_cluster = min(
                range(cluster_count),
                key=lambda cluster_index: (x_value - centers[cluster_index][0]) ** 2
                + (y_value - centers[cluster_index][1]) ** 2,
            )
            if assignments[index] != best_cluster:
                assignments[index] = best_cluster
                changed = True

        totals = [[0.0, 0.0, 0] for _ in range(cluster_count)]
        for (x_value, y_value), cluster_index in zip(points, assignments):
            totals[cluster_index][0] += x_value
            totals[cluster_index][1] += y_value
            totals[cluster_index][2] += 1

        for cluster_index, (x_total, y_total, count) in enumerate(totals):
            if count:
                centers[cluster_index] = [x_total / count, y_total / count]

        if not changed:
            break

    return assignments, [(center[0], center[1]) for center in centers]


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Missing database: {DB_PATH}")

    random.seed(RANDOM_SEED)
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    rows = list(
        cursor.execute(
            """
            SELECT m.video_id, m.seg_index, m.en, m.zh, v.vector
            FROM tm_main AS m
            JOIN tm_vectors AS v USING (content_sha)
            WHERE v.model_id = ?
            ORDER BY m.video_id, m.seg_index
            """,
            (MODEL_ID,),
        )
    )
    connection.close()

    if not rows:
        raise SystemExit("No semantic rows found.")

    vector_dim = len(rows[0][4]) // 4
    vectors = array("f")
    mean = [0.0] * vector_dim
    metadata: list[dict[str, object]] = []

    for video_id, seg_index, en, zh, blob in rows:
        unpacked = struct.unpack(f"<{vector_dim}f", blob)
        vectors.extend(unpacked)
        metadata.append(
            {
                "entryId": f"{video_id}#{seg_index}",
                "videoId": video_id,
                "segIndex": seg_index,
                "en": en,
                "zh": zh,
            }
        )

        for dimension, value in enumerate(unpacked):
            mean[dimension] += value

    entry_count = len(metadata)
    mean = [value / entry_count for value in mean]

    def centered_dot(row_index: int, direction: list[float]) -> float:
        total = 0.0
        offset = row_index * vector_dim
        for dimension in range(vector_dim):
            total += (vectors[offset + dimension] - mean[dimension]) * direction[dimension]
        return total

    principal_components: list[list[float]] = []
    for _ in range(2):
        direction = normalize([random.random() - 0.5 for _ in range(vector_dim)])

        for _ in range(PCA_ITERATIONS):
            updated = [0.0] * vector_dim
            for row_index in range(entry_count):
                score = centered_dot(row_index, direction)
                offset = row_index * vector_dim
                for dimension in range(vector_dim):
                    updated[dimension] += score * (vectors[offset + dimension] - mean[dimension])

            for previous in principal_components:
                projection = sum(value * basis for value, basis in zip(updated, previous))
                for dimension in range(vector_dim):
                    updated[dimension] -= projection * previous[dimension]

            direction = normalize(updated)

        principal_components.append(direction)

    coordinates: list[tuple[float, float]] = []
    x_values: list[float] = []
    y_values: list[float] = []
    for row_index in range(entry_count):
        x_value = centered_dot(row_index, principal_components[0])
        y_value = centered_dot(row_index, principal_components[1])
        coordinates.append((x_value, y_value))
        x_values.append(x_value)
        y_values.append(y_value)

    x_low, x_high = percentile_bounds(x_values)
    y_low, y_high = percentile_bounds(y_values)
    scaled_points = [
        (scale_value(x_value, x_low, x_high), scale_value(y_value, y_low, y_high))
        for x_value, y_value in coordinates
    ]

    assignments, cluster_centers = kmeans_2d(
        [(x_value / 1000, y_value / 1000) for x_value, y_value in scaled_points],
        CLUSTER_COUNT,
    )

    global_counts: Counter[str] = Counter()
    cluster_counts: dict[int, Counter[str]] = defaultdict(Counter)
    cluster_members: dict[int, list[int]] = defaultdict(list)

    for index, item in enumerate(metadata):
        cluster_id = assignments[index]
        cluster_members[cluster_id].append(index)
        tokens = [
            token
            for token in TOKEN_RE.findall(str(item["en"]).lower())
            if len(token) > 2 and token not in STOPWORDS
        ]
        global_counts.update(tokens)
        cluster_counts[cluster_id].update(tokens)

    clusters = []
    for cluster_id in range(CLUSTER_COUNT):
        members = cluster_members[cluster_id]
        keyword_scores = []
        for token, count in cluster_counts[cluster_id].items():
            if count < 5:
                continue
            keyword_scores.append((count / max(1, global_counts[token]), count, token))
        keyword_scores.sort(reverse=True)
        keywords = [token for _, _, token in keyword_scores[:5]]

        center_x = cluster_centers[cluster_id][0] * 1000
        center_y = cluster_centers[cluster_id][1] * 1000
        representative = sorted(
            members,
            key=lambda index: (scaled_points[index][0] - center_x) ** 2 + (scaled_points[index][1] - center_y) ** 2,
        )

        samples = []
        seen = set()
        for index in representative:
            item = metadata[index]
            text_key = str(item["en"])
            if text_key in seen:
                continue
            seen.add(text_key)
            samples.append(
                {
                    "entryId": item["entryId"],
                    "videoId": item["videoId"],
                    "segIndex": item["segIndex"],
                    "en": item["en"],
                    "zh": item["zh"],
                }
            )
            if len(samples) == 3:
                break

        clusters.append(
            {
                "id": cluster_id,
                "label": f"Cluster {cluster_id + 1}",
                "color": PALETTE[cluster_id % len(PALETTE)],
                "size": len(members),
                "x": round(center_x),
                "y": round(center_y),
                "keywords": keywords,
                "samples": samples,
            }
        )

    points = []
    for index, item in enumerate(metadata):
        points.append(
            {
                "entryId": item["entryId"],
                "videoId": item["videoId"],
                "segIndex": item["segIndex"],
                "en": item["en"],
                "zh": item["zh"],
                "x": scaled_points[index][0],
                "y": scaled_points[index][1],
                "clusterId": assignments[index],
            }
        )

    payload = {
        "version": 1,
        "projection": "pca-2d",
        "clusterAlgorithm": "kmeans-2d",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceDb": DB_PATH.name,
        "modelId": MODEL_ID,
        "pointCount": entry_count,
        "vectorDim": vector_dim,
        "clusters": clusters,
        "points": points,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH} with {entry_count} points.")


if __name__ == "__main__":
    main()
