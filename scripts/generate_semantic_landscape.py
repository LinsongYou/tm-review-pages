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
FLOW_BIN_COUNT = 24
FINGERPRINT_BIN_COUNT = 32
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a",
    "about",
    "actually",
    "after",
    "again",
    "all",
    "almost",
    "also",
    "am",
    "an",
    "and",
    "any",
    "anyway",
    "are",
    "around",
    "as",
    "at",
    "back",
    "basically",
    "be",
    "because",
    "been",
    "being",
    "best",
    "bit",
    "but",
    "by",
    "can",
    "car",
    "cars",
    "come",
    "could",
    "day",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "dont",
    "down",
    "drive",
    "driving",
    "even",
    "every",
    "first",
    "for",
    "from",
    "get",
    "getting",
    "go",
    "going",
    "gone",
    "gonna",
    "good",
    "got",
    "great",
    "had",
    "has",
    "have",
    "having",
    "he",
    "hello",
    "here",
    "how",
    "i",
    "if",
    "im",
    "in",
    "into",
    "is",
    "it",
    "its",
    "ive",
    "just",
    "kind",
    "kinds",
    "know",
    "last",
    "lets",
    "like",
    "little",
    "look",
    "lot",
    "lots",
    "made",
    "many",
    "maybe",
    "me",
    "mean",
    "more",
    "most",
    "much",
    "my",
    "need",
    "never",
    "nice",
    "no",
    "not",
    "now",
    "of",
    "off",
    "oh",
    "okay",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "pretty",
    "probably",
    "quite",
    "really",
    "right",
    "said",
    "say",
    "see",
    "she",
    "should",
    "so",
    "some",
    "something",
    "still",
    "such",
    "than",
    "thank",
    "thanks",
    "that",
    "thats",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "think",
    "this",
    "those",
    "three",
    "through",
    "time",
    "to",
    "too",
    "two",
    "up",
    "us",
    "very",
    "was",
    "way",
    "we",
    "well",
    "went",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "yeah",
    "yes",
    "you",
    "your",
    "let",
    "s",
}
GENERIC_LABEL_TOKENS = {
    "awesome",
    "book",
    "books",
    "cool",
    "day",
    "great",
    "hello",
    "loved",
    "merino",
    "nice",
    "okay",
    "russian",
    "statesidesupercars",
    "tuned",
    "video",
    "wool",
    "write",
}
LABEL_CANONICAL = {
    "batteries": "Battery",
    "battery": "Battery",
    "book": "Book",
    "books": "Books",
    "brake": "Brake",
    "brakes": "Brakes",
    "caliper": "Caliper",
    "calipers": "Calipers",
    "charging": "Charging",
    "chassis": "Chassis",
    "conditions": "Conditions",
    "feedback": "Feedback",
    "gearbox": "Gearbox",
    "grip": "Grip",
    "lap": "Lap",
    "laps": "Laps",
    "music": "Music",
    "outro": "Outro",
    "playlist": "Playlist",
    "power": "Power",
    "range": "Range",
    "record": "Record",
    "speed": "Speed",
    "suspension": "Suspension",
    "throttle": "Throttle",
    "track": "Track",
    "tyre": "Tyre",
    "tyres": "Tyres",
    "wheel": "Wheel",
    "wheels": "Wheels",
}
THEME_RULES = [
    (
        "Brakes, Grip & Chassis",
        {
            "aero",
            "brake",
            "brakes",
            "braking",
            "caliper",
            "calipers",
            "carousel",
            "chassis",
            "compliance",
            "cup tyres",
            "grip",
            "handling",
            "pressure",
            "suspension",
            "traction",
            "tyre",
            "tyre pressure",
            "tyres",
            "wheel",
            "wheelbase",
            "wheels",
        },
    ),
    (
        "Suspension & Front-End Setup",
        {
            "camber",
            "compression",
            "dampers",
            "front",
            "front end",
            "oil",
            "setup",
            "splitter",
            "suspension",
        },
    ),
    (
        "Powertrain, Charging & Range",
        {
            "battery",
            "charge",
            "charging",
            "electric",
            "engine",
            "engines",
            "gearbox",
            "horsepower",
            "kilowatts",
            "motor",
            "power",
            "range",
            "rpm",
            "torque",
        },
    ),
    (
        "Lap Times & Conditions",
        {
            "conditions",
            "lap",
            "lap time",
            "laps",
            "minutes",
            "rain",
            "record",
            "seconds",
            "speed",
            "timing",
            "track conditions",
            "wet",
        },
    ),
    (
        "Track Access & Logistics",
        {
            "closed",
            "entry",
            "logistics",
            "marshal",
            "overtake",
            "overtakes",
            "parking",
            "pass",
            "passengers",
            "rental",
            "rent",
            "road",
            "roads",
            "session",
            "track",
            "track day",
            "traffic",
        },
    ),
    (
        "Road Speed & Usability",
        {
            "daily",
            "highway",
            "kilometers",
            "license",
            "license plate",
            "per hour",
            "plate",
            "road",
            "speed",
            "street",
        },
    ),
    (
        "Driving Technique & Feedback",
        {
            "balance",
            "bumps",
            "corner",
            "corners",
            "feedback",
            "full throttle",
            "heel",
            "line",
            "rolling",
            "speedometer",
            "steer",
            "steering",
            "throttle",
            "trail",
            "transition",
            "turn",
            "turning",
        },
    ),
    (
        "Build Plans & Hardware",
        {
            "aero",
            "build",
            "carbon",
            "cage",
            "fiber",
            "heavier",
            "lighter",
            "parts",
            "strip",
            "stripped",
            "upgrade",
            "weight",
            "wing",
        },
    ),
    (
        "Mods, Styling & Materials",
        {
            "carpets",
            "color",
            "interior",
            "looks",
            "material",
            "materials",
            "merino",
            "mods",
            "rebuild",
            "styling",
            "wool",
            "wrap",
            "wrapped",
        },
    ),
    (
        "Video Production & Outro",
        {
            "book",
            "books",
            "filming",
            "music",
            "outro",
            "playlist",
            "recording",
            "share",
            "subscribe",
            "tomorrow",
            "video",
            "youtube",
        },
    ),
    (
        "Positive Reactions & Thanks",
        {
            "alright",
            "alrighty",
            "amazing",
            "enjoyed",
            "hope enjoyed",
            "welcome",
        },
    ),
    (
        "Short Reactions & Fillers",
        {
            "alright",
            "amazing",
            "awesome",
            "banter",
            "crazy",
            "fun",
            "insane",
            "joke",
            "laugh",
            "loved",
            "nice",
            "welcome",
            "wow",
        },
    ),
    (
        "Group Banter & Setups",
        {
            "follow",
            "fun",
            "line up",
            "passenger",
            "passengers",
            "setup",
            "show",
            "take",
            "try",
        },
    ),
]
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


def normalize_token(token: str) -> str:
    normalized = token.lower().strip("'")
    for suffix in ("'s", "'re", "'ve", "'ll", "'m", "'d", "n't"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text):
        normalized = normalize_token(token)
        if normalized:
            tokens.append(normalized)
    return tokens


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    ]


def extract_keyphrases(text: str) -> list[str]:
    tokens = content_tokens(text)
    phrases: list[str] = []

    for token in tokens:
        phrases.append(token)

    for index in range(len(tokens) - 1):
        left = tokens[index]
        right = tokens[index + 1]
        if left == right:
            continue
        phrases.append(f"{left} {right}")

    return phrases


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


def to_bin_index(value: float, bin_count: int) -> int:
    clamped = min(0.999999, max(0.0, value))
    return min(bin_count - 1, max(0, int(clamped * bin_count)))


def format_progress_label(start: float, end: float) -> str:
    start_percent = round(start * 100)
    end_percent = round(end * 100)
    if start_percent == end_percent:
        return f"{start_percent}%"
    return f"{start_percent}% - {end_percent}%"


def titleize_keyword(keyword: str) -> str:
    words = [LABEL_CANONICAL.get(word, word.title()) for word in keyword.split()]
    return " ".join(words)


def build_fallback_label(keywords: list[str], used_labels: set[str]) -> str:
    parts: list[str] = []

    for keyword in keywords:
        if keyword in GENERIC_LABEL_TOKENS:
            continue
        titled = titleize_keyword(keyword)
        if not titled or titled in parts:
            continue
        parts.append(titled)
        if len(parts) == 2:
            break

    if len(parts) >= 2:
        candidate = f"{parts[0]} & {parts[1]}"
    elif parts:
        candidate = parts[0]
    else:
        candidate = "Semantic Region"

    if candidate not in used_labels:
        return candidate

    suffix = 2
    while f"{candidate} {suffix}" in used_labels:
        suffix += 1
    return f"{candidate} {suffix}"


def score_theme_labels(keywords: list[str], samples: list[dict[str, object]]) -> list[tuple[float, str]]:
    weighted_terms: Counter[str] = Counter()

    for rank, keyword in enumerate(keywords[:6]):
        weight = max(1.0, 6.0 - rank)
        weighted_terms[keyword] += weight * 1.4
        for token in tokenize(keyword):
            if token not in STOPWORDS:
                weighted_terms[token] += weight

    for sample in samples:
        for token in content_tokens(str(sample["en"])):
            weighted_terms[token] += 0.4

    scored: list[tuple[float, str]] = []
    for label, hints in THEME_RULES:
        score = 0.0
        for term, weight in weighted_terms.items():
            if term in hints:
                score += weight * (1.3 if " " in term else 1.0)
                continue

            token_overlap = len(set(term.split()).intersection(hints))
            if token_overlap:
                score += weight * token_overlap * 0.55

        if score > 0:
            scored.append((score, label))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def infer_cluster_label(
    keywords: list[str],
    samples: list[dict[str, object]],
    used_labels: set[str],
) -> str:
    for score, label in score_theme_labels(keywords, samples):
        if score >= 5.5 and label not in used_labels:
            return label

    return build_fallback_label(keywords, used_labels)


def build_cluster_keyword_scores(
    metadata: list[dict[str, object]],
    assignments: list[int],
) -> dict[int, list[str]]:
    global_counts: Counter[str] = Counter()
    cluster_counts: dict[int, Counter[str]] = defaultdict(Counter)
    global_video_support: defaultdict[str, set[str]] = defaultdict(set)
    cluster_video_support: dict[int, defaultdict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for index, item in enumerate(metadata):
        cluster_id = assignments[index]
        video_id = str(item["videoId"])
        phrases = extract_keyphrases(str(item["en"]))
        if not phrases:
            continue

        global_counts.update(phrases)
        cluster_counts[cluster_id].update(phrases)

        for phrase in set(phrases):
            global_video_support[phrase].add(video_id)
            cluster_video_support[cluster_id][phrase].add(video_id)

    cluster_keywords: dict[int, list[str]] = {}
    for cluster_id in range(CLUSTER_COUNT):
        ranked: list[tuple[float, str]] = []
        counts = cluster_counts[cluster_id]
        total_cluster_videos = max(1, len({str(metadata[index]["videoId"]) for index, assigned in enumerate(assignments) if assigned == cluster_id}))

        for phrase, count in counts.items():
            phrase_word_count = phrase.count(" ") + 1
            min_count = 3 if phrase_word_count == 1 else 2
            if count < min_count:
                continue

            global_count = global_counts[phrase]
            if global_count <= 0:
                continue

            video_support = len(cluster_video_support[cluster_id][phrase])
            global_video_count = len(global_video_support[phrase])
            if video_support < 2 or global_video_count <= 0:
                continue

            if phrase in GENERIC_LABEL_TOKENS:
                continue

            specificity = count / global_count
            video_specificity = video_support / global_video_count
            coverage = video_support / total_cluster_videos
            phrase_bonus = 1.18 if phrase_word_count > 1 else 1.0
            score = (count ** 1.05) * specificity * (1 + video_specificity) * (1 + coverage) * phrase_bonus
            ranked.append((score, phrase))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        cluster_keywords[cluster_id] = [phrase for _, phrase in ranked[:6]]

    return cluster_keywords


def build_semantic_flow(
    metadata: list[dict[str, object]],
    assignments: list[int],
) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        grouped[str(item["videoId"])].append(index)

    cluster_counts = [[0 for _ in range(CLUSTER_COUNT)] for _ in range(FLOW_BIN_COUNT)]
    totals = [0 for _ in range(FLOW_BIN_COUNT)]

    for indexes in grouped.values():
        starts = [
            int(metadata[index]["startMs"])
            for index in indexes
            if metadata[index]["startMs"] is not None and metadata[index]["endMs"] is not None
        ]
        ends = [
            int(metadata[index]["endMs"])
            for index in indexes
            if metadata[index]["startMs"] is not None and metadata[index]["endMs"] is not None
        ]

        if not starts or not ends:
            continue

        video_start = min(starts)
        video_end = max(ends)
        span = max(1, video_end - video_start)

        for index in indexes:
            start_ms = metadata[index]["startMs"]
            end_ms = metadata[index]["endMs"]
            if start_ms is None or end_ms is None or end_ms <= start_ms:
                continue

            midpoint = ((int(start_ms) + int(end_ms)) / 2 - video_start) / span
            bin_index = to_bin_index(midpoint, FLOW_BIN_COUNT)
            cluster_id = assignments[index]
            cluster_counts[bin_index][cluster_id] += 1
            totals[bin_index] += 1

    peak_total = max(totals) if totals else 0
    peak_bin_index = totals.index(peak_total) if peak_total else 0
    bins: list[dict[str, object]] = []

    leading_cluster_id = -1
    trailing_cluster_id = -1
    for bin_index, total in enumerate(totals):
        dominant_cluster_id = -1
        if total > 0:
            dominant_cluster_id = max(
                range(CLUSTER_COUNT),
                key=lambda cluster_id: (cluster_counts[bin_index][cluster_id], -cluster_id),
            )
            if leading_cluster_id < 0:
                leading_cluster_id = dominant_cluster_id
            trailing_cluster_id = dominant_cluster_id

        start = bin_index / FLOW_BIN_COUNT
        end = (bin_index + 1) / FLOW_BIN_COUNT
        bins.append(
            {
                "start": start,
                "end": end,
                "label": format_progress_label(start, end),
                "total": total,
                "clusterCounts": cluster_counts[bin_index],
                "dominantClusterId": dominant_cluster_id,
            }
        )

    return {
        "binCount": FLOW_BIN_COUNT,
        "peakBinIndex": peak_bin_index,
        "peakTotal": peak_total,
        "leadingClusterId": leading_cluster_id,
        "trailingClusterId": trailing_cluster_id,
        "bins": bins,
    }


def build_video_fingerprint_wall(
    metadata: list[dict[str, object]],
    assignments: list[int],
    scaled_points: list[tuple[int, int]],
) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        grouped[str(item["videoId"])].append(index)

    videos: list[dict[str, object]] = []
    for video_id, indexes in grouped.items():
        starts = [
            int(metadata[index]["startMs"])
            for index in indexes
            if metadata[index]["startMs"] is not None and metadata[index]["endMs"] is not None
        ]
        ends = [
            int(metadata[index]["endMs"])
            for index in indexes
            if metadata[index]["startMs"] is not None and metadata[index]["endMs"] is not None
        ]

        if starts and ends:
            video_start = min(starts)
            video_end = max(ends)
        else:
            video_start = 0
            video_end = max(1, len(indexes))

        span = max(1, video_end - video_start)
        density_counts = [0 for _ in range(FINGERPRINT_BIN_COUNT)]
        cluster_bin_counts = [Counter() for _ in range(FINGERPRINT_BIN_COUNT)]
        cluster_counts = [0 for _ in range(CLUSTER_COUNT)]
        timed_entry_count = 0
        x_total = 0
        y_total = 0

        ordered_indexes = sorted(indexes, key=lambda index: int(metadata[index]["segIndex"]))
        first_entry_id = str(metadata[ordered_indexes[0]]["entryId"]) if ordered_indexes else ""

        for index in ordered_indexes:
            cluster_id = assignments[index]
            cluster_counts[cluster_id] += 1
            x_total += scaled_points[index][0]
            y_total += scaled_points[index][1]

            start_ms = metadata[index]["startMs"]
            end_ms = metadata[index]["endMs"]
            if start_ms is None or end_ms is None or end_ms <= start_ms:
                continue

            timed_entry_count += 1
            midpoint = ((int(start_ms) + int(end_ms)) / 2 - video_start) / span
            bin_index = to_bin_index(midpoint, FINGERPRINT_BIN_COUNT)
            density_counts[bin_index] += 1
            cluster_bin_counts[bin_index][cluster_id] += 1

        max_density = max(density_counts) if density_counts else 0
        densities: list[float] = []
        bins: list[int] = []
        for bin_index in range(FINGERPRINT_BIN_COUNT):
            density = density_counts[bin_index]
            densities.append(round(density / max_density, 4) if max_density else 0.0)
            if density == 0:
                bins.append(-1)
                continue
            dominant_cluster_id, _ = max(
                cluster_bin_counts[bin_index].items(),
                key=lambda item: (item[1], -item[0]),
            )
            bins.append(dominant_cluster_id)

        dominant_cluster_id = max(
            range(CLUSTER_COUNT),
            key=lambda cluster_id: (cluster_counts[cluster_id], -cluster_id),
        )
        entry_count = len(ordered_indexes)
        videos.append(
            {
                "videoId": video_id,
                "firstEntryId": first_entry_id,
                "entryCount": entry_count,
                "timedEntryCount": timed_entry_count,
                "dominantClusterId": dominant_cluster_id,
                "dominantShare": round(cluster_counts[dominant_cluster_id] / entry_count, 4) if entry_count else 0.0,
                "clusterCounts": cluster_counts,
                "bins": bins,
                "densities": densities,
                "x": round(x_total / entry_count) if entry_count else 500,
                "y": round(y_total / entry_count) if entry_count else 500,
            }
        )

    videos.sort(key=lambda item: (int(item["x"]), int(item["y"]), -int(item["entryCount"]), str(item["videoId"])))

    return {
        "binCount": FINGERPRINT_BIN_COUNT,
        "sort": "projection-x",
        "videos": videos,
    }


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Missing database: {DB_PATH}")

    random.seed(RANDOM_SEED)
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    rows = list(
        cursor.execute(
            """
            SELECT m.video_id, m.seg_index, m.en, m.zh, m.start_ms, m.end_ms, v.vector
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

    vector_dim = len(rows[0][6]) // 4
    vectors = array("f")
    mean = [0.0] * vector_dim
    metadata: list[dict[str, object]] = []

    for video_id, seg_index, en, zh, start_ms, end_ms, blob in rows:
        unpacked = struct.unpack(f"<{vector_dim}f", blob)
        vectors.extend(unpacked)
        metadata.append(
            {
                "entryId": f"{video_id}#{seg_index}",
                "videoId": video_id,
                "segIndex": seg_index,
                "en": en,
                "zh": zh,
                "startMs": start_ms,
                "endMs": end_ms,
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

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(assignments):
        cluster_members[cluster_id].append(index)

    cluster_keywords = build_cluster_keyword_scores(metadata, assignments)

    clusters_by_id: dict[int, dict[str, object]] = {}
    used_labels: set[str] = set()
    cluster_order = sorted(range(CLUSTER_COUNT), key=lambda cluster_id: (-len(cluster_members[cluster_id]), cluster_id))
    for cluster_id in cluster_order:
        members = cluster_members[cluster_id]
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
            if len(samples) == 4:
                break

        keywords = cluster_keywords.get(cluster_id, [])
        label = infer_cluster_label(keywords, samples, used_labels)
        used_labels.add(label)

        clusters_by_id[cluster_id] = {
            "id": cluster_id,
            "label": label,
            "color": PALETTE[cluster_id % len(PALETTE)],
            "size": len(members),
            "x": round(center_x),
            "y": round(center_y),
            "keywords": keywords,
            "samples": samples,
        }

    clusters = [clusters_by_id[cluster_id] for cluster_id in range(CLUSTER_COUNT)]

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
        "version": 2,
        "projection": "pca-2d",
        "clusterAlgorithm": "kmeans-2d",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceDb": DB_PATH.name,
        "modelId": MODEL_ID,
        "pointCount": entry_count,
        "vectorDim": vector_dim,
        "clusters": clusters,
        "points": points,
        "semanticFlowTimeline": build_semantic_flow(metadata, assignments),
        "videoFingerprintWall": build_video_fingerprint_wall(metadata, assignments, scaled_points),
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH} with {entry_count} points.")


if __name__ == "__main__":
    main()
