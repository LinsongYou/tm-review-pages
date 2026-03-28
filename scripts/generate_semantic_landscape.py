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
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "startup-visualizations.json"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_COUNT = 10
PCA_ITERATIONS = 18
CLUSTER_ITERATIONS = 14
RANDOM_SEED = 7
PERCENTILE_CLIP = 0.02
FINGERPRINT_BIN_COUNT = 32
MEDOID_CANDIDATE_COUNT = 12
REPRESENTATIVE_POOL_COUNT = 48
REPRESENTATIVE_SAMPLE_COUNT = 4
THEME_SCORE_THRESHOLD = 8.0
THEME_MARGIN_THRESHOLD = 2.4
PROVISIONAL_THEME_SCORE_THRESHOLD = 4.5
PROVISIONAL_THEME_MARGIN_THRESHOLD = 0.35
HIGH_CONFIDENCE_LABEL_THRESHOLD = 0.84
HIGH_CONFIDENCE_VIDEO_MIN = 3
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
    "bye",
    "code",
    "construction",
    "cool",
    "couple mods",
    "curb",
    "day",
    "definitely",
    "enjoy",
    "enjoy lap",
    "everything",
    "flat",
    "great",
    "guys",
    "guys enjoy",
    "happen",
    "hello",
    "hope enjoyed",
    "interesting",
    "licensed",
    "licensed products",
    "loved",
    "man",
    "merino",
    "nice",
    "okay",
    "rburgring licensed",
    "russian",
    "sad",
    "second lap",
    "statesidesupercars",
    "stock",
    "talk",
    "tuned",
    "use code",
    "video",
    "wool",
    "write",
    "yellow",
}
LABEL_CANONICAL = {
    "abs": "ABS",
    "brake": "Brake",
    "brakes": "Brakes",
    "chassis": "Chassis",
    "corner": "Corners",
    "corners": "Corners",
    "engine": "Engine",
    "euros": "Euros",
    "flag": "Flags",
    "flags": "Flags",
    "gearbox": "Gearbox",
    "grip": "Grip",
    "gt3": "GT3",
    "hour": "Hour",
    "kilometers": "Kilometers",
    "kilos": "Kilos",
    "lap": "Lap",
    "laps": "Laps",
    "people": "People",
    "price": "Price",
    "prices": "Prices",
    "speed": "Speed",
    "suspension": "Suspension",
    "turbo": "Turbo",
    "wet": "Wet",
    "yep": "Short Replies",
}
THEME_RULES = [
    (
        "Brakes, Grip & Chassis",
        {
            "brake",
            "brakes",
            "braking",
            "caliper",
            "calipers",
            "chassis",
            "cup tyres",
            "grip",
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
            "front end",
            "ride height",
            "splitter",
            "suspension",
        },
    ),
    (
        "Powertrain & Performance",
        {
            "battery",
            "charging",
            "electric",
            "engine",
            "engines",
            "ethanol",
            "gearbox",
            "horsepower",
            "kilowatts",
            "motor",
            "power",
            "range",
            "rpm",
            "torque",
            "turbo",
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
            "marshal",
            "parking",
            "passenger",
            "passengers",
            "session",
            "track day",
            "traffic",
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
            "steering",
            "throttle",
            "trail",
            "transition",
            "turn",
            "turning",
        },
    ),
    (
        "Suspension, Grip & Speed",
        {
            "abs",
            "downforce",
            "front",
            "grip",
            "high speed",
            "speed",
            "suspension",
            "weight",
        },
    ),
    (
        "Corners, Flags & Wet Conditions",
        {
            "corner",
            "corners",
            "flag",
            "left",
            "umbrella",
            "wet",
            "yellow flag",
        },
    ),
    (
        "Build Plans & Hardware",
        {
            "aero",
            "build",
            "cage",
            "carbon",
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
        "Styling & Interior Materials",
        {
            "carpets",
            "color",
            "interior",
            "looks",
            "material",
            "materials",
            "merino",
            "styling",
            "wool",
            "wrap",
            "wrapped",
        },
    ),
    (
        "Video Production & Outro",
        {
            "filming",
            "music",
            "outro",
            "playlist",
            "recording",
            "share",
            "subscribe",
            "video",
            "youtube",
        },
    ),
    (
        "Positive Reactions & Thanks",
        {
            "amazing",
            "appreciate",
            "enjoyed",
            "loved",
            "thank",
            "thanks",
            "welcome",
        },
    ),
    (
        "Short Reactions & Acknowledgements",
        {
            "alright",
            "alrighty",
            "nein",
            "worries",
            "yep",
        },
    ),
    (
        "Starts, Passes & Next Moves",
        {
            "ahead",
            "go",
            "hopefully",
            "next",
            "pass",
            "take",
            "take easy",
            "watch",
        },
    ),
    (
        "People, Comments & Mentions",
        {
            "comments",
            "else",
            "mention",
            "mods",
            "people",
            "reason",
            "things",
            "years",
        },
    ),
    (
        "Speeds, Prices & Numbers",
        {
            "euros",
            "hour",
            "kilometers",
            "kilometers per",
            "kilos",
            "per hour",
            "price",
        },
    ),
    (
        "Group Banter & Setups",
        {
            "banter",
            "follow",
            "line up",
            "passenger",
            "passengers",
            "setup",
            "try",
        },
    ),
]
THEME_HINTS_BY_LABEL = {label: hints for label, hints in THEME_RULES}
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


def normalize(vector: list[float] | tuple[float, ...]) -> list[float]:
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


def row_offset(row_index: int, vector_dim: int) -> int:
    return row_index * vector_dim


def copy_row(vectors: array, row_index: int, vector_dim: int) -> list[float]:
    offset = row_offset(row_index, vector_dim)
    return [vectors[offset + dimension] for dimension in range(vector_dim)]


def dot_row_to_vector(vectors: array, row_index: int, vector_dim: int, other: list[float]) -> float:
    total = 0.0
    offset = row_offset(row_index, vector_dim)
    for dimension in range(vector_dim):
        total += vectors[offset + dimension] * other[dimension]
    return total


def dot_rows(vectors: array, left_index: int, right_index: int, vector_dim: int) -> float:
    total = 0.0
    left_offset = row_offset(left_index, vector_dim)
    right_offset = row_offset(right_index, vector_dim)
    for dimension in range(vector_dim):
        total += vectors[left_offset + dimension] * vectors[right_offset + dimension]
    return total


def add_row(target: list[float], vectors: array, row_index: int, vector_dim: int) -> None:
    offset = row_offset(row_index, vector_dim)
    for dimension in range(vector_dim):
        target[dimension] += vectors[offset + dimension]


def initialize_cluster_centers(
    vectors: array,
    entry_count: int,
    vector_dim: int,
    cluster_count: int,
) -> list[list[float]]:
    random_generator = random.Random(RANDOM_SEED)
    first_index = random_generator.randrange(entry_count)
    centers = [copy_row(vectors, first_index, vector_dim)]
    nearest_distances = [0.0 for _ in range(entry_count)]

    for row_index in range(entry_count):
        similarity = dot_row_to_vector(vectors, row_index, vector_dim, centers[0])
        nearest_distances[row_index] = max(0.0, 1.0 - similarity)

    while len(centers) < cluster_count:
        total_distance = sum(nearest_distances)
        if total_distance <= 1e-9:
            fallback_index = len(centers) % entry_count
            centers.append(copy_row(vectors, fallback_index, vector_dim))
            continue

        threshold = random_generator.random() * total_distance
        running = 0.0
        next_index = entry_count - 1
        for row_index, distance in enumerate(nearest_distances):
            running += distance
            if running >= threshold:
                next_index = row_index
                break

        centers.append(copy_row(vectors, next_index, vector_dim))
        latest_center = centers[-1]
        for row_index in range(entry_count):
            similarity = dot_row_to_vector(vectors, row_index, vector_dim, latest_center)
            distance = max(0.0, 1.0 - similarity)
            if distance < nearest_distances[row_index]:
                nearest_distances[row_index] = distance

    return centers


def spherical_kmeans(
    vectors: array,
    entry_count: int,
    vector_dim: int,
    cluster_count: int,
) -> tuple[list[int], list[list[float]]]:
    centers = initialize_cluster_centers(vectors, entry_count, vector_dim, cluster_count)
    assignments = [-1] * entry_count

    for _ in range(CLUSTER_ITERATIONS):
        cluster_sums = [[0.0] * vector_dim for _ in range(cluster_count)]
        cluster_counts = [0] * cluster_count
        best_scores = [-1.0] * entry_count
        changed = False

        for row_index in range(entry_count):
            best_cluster = 0
            best_score = -float("inf")

            for cluster_id, center in enumerate(centers):
                score = dot_row_to_vector(vectors, row_index, vector_dim, center)
                if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12 and cluster_id < best_cluster
                ):
                    best_cluster = cluster_id
                    best_score = score

            if assignments[row_index] != best_cluster:
                assignments[row_index] = best_cluster
                changed = True

            best_scores[row_index] = best_score
            cluster_counts[best_cluster] += 1
            add_row(cluster_sums[best_cluster], vectors, row_index, vector_dim)

        empty_clusters = [cluster_id for cluster_id, count in enumerate(cluster_counts) if count == 0]
        if empty_clusters:
            farthest_rows = sorted(
                range(entry_count),
                key=lambda row_index: (best_scores[row_index], row_index),
            )
            used_rows: set[int] = set()
            for cluster_id, row_index in zip(empty_clusters, farthest_rows):
                while row_index in used_rows:
                    row_index += 1
                    if row_index >= entry_count:
                        row_index = 0
                centers[cluster_id] = copy_row(vectors, row_index, vector_dim)
                used_rows.add(row_index)
        else:
            for cluster_id in range(cluster_count):
                centers[cluster_id] = normalize(cluster_sums[cluster_id])

        if not changed and not empty_clusters:
            break

    return assignments, centers


def to_bin_index(value: float, bin_count: int) -> int:
    clamped = min(0.999999, max(0.0, value))
    return min(bin_count - 1, max(0, int(clamped * bin_count)))


def build_cluster_phrase_scores(
    metadata: list[dict[str, object]],
    assignments: list[int],
) -> dict[int, list[tuple[float, str, int, int]]]:
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

    cluster_phrases: dict[int, list[tuple[float, str, int, int]]] = {}
    for cluster_id in range(CLUSTER_COUNT):
        ranked: list[tuple[float, str, int, int]] = []
        counts = cluster_counts[cluster_id]
        total_cluster_videos = max(
            1,
            len(
                {
                    str(metadata[index]["videoId"])
                    for index, assigned in enumerate(assignments)
                    if assigned == cluster_id
                }
            ),
        )

        for phrase, count in counts.items():
            phrase_word_count = phrase.count(" ") + 1
            min_count = 3 if phrase_word_count == 1 else 2
            if count < min_count:
                continue

            global_count = global_counts[phrase]
            if global_count <= 0 or phrase in GENERIC_LABEL_TOKENS:
                continue

            video_support = len(cluster_video_support[cluster_id][phrase])
            global_video_count = len(global_video_support[phrase])
            if video_support < 2 or global_video_count <= 0:
                continue

            specificity = count / global_count
            video_specificity = video_support / global_video_count
            coverage = video_support / total_cluster_videos
            phrase_bonus = 1.22 if phrase_word_count > 1 else 1.0
            score = (count ** 1.05) * specificity * (1 + video_specificity) * (0.8 + coverage) * phrase_bonus
            ranked.append((score, phrase, count, video_support))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        cluster_phrases[cluster_id] = ranked[:8]

    return cluster_phrases


def label_matches_theme(phrase: str, hints: set[str]) -> bool:
    if phrase in hints:
        return True

    phrase_tokens = set(tokenize(phrase))
    if not phrase_tokens:
        return False

    return bool(phrase_tokens.intersection(hints))


def count_theme_support(phrases: list[str], hints: set[str]) -> int:
    return sum(1 for phrase in phrases if label_matches_theme(phrase, hints))


def titleize_phrase(phrase: str) -> str:
    words = [LABEL_CANONICAL.get(word, word.title()) for word in phrase.split()]
    return " ".join(words)


def build_fallback_label(
    cluster_id: int,
    phrase_scores: list[tuple[float, str, int, int]],
) -> str:
    parts: list[str] = []

    for _, phrase, _, _ in phrase_scores:
        if phrase in GENERIC_LABEL_TOKENS:
            continue

        titled = titleize_phrase(phrase)
        if not titled or titled in parts:
            continue

        parts.append(titled)
        if len(parts) == 2:
            break

    if len(parts) >= 2:
        return f"{parts[0]} & {parts[1]}"
    if parts:
        return parts[0]
    return f"Region {cluster_id + 1}"


def make_unique_label(label: str, used_labels: set[str]) -> str:
    if label not in used_labels:
        used_labels.add(label)
        return label

    suffix = 2
    while f"{label} {suffix}" in used_labels:
        suffix += 1

    unique_label = f"{label} {suffix}"
    used_labels.add(unique_label)
    return unique_label


def score_theme_labels(
    phrase_scores: list[tuple[float, str, int, int]],
    samples: list[dict[str, object]],
) -> list[tuple[float, str]]:
    weighted_terms: Counter[str] = Counter()

    for rank, (score, phrase, _, _) in enumerate(phrase_scores[:8]):
        weight = max(1.0, min(5.0, score / 2.0))
        weighted_terms[phrase] += weight * 1.45

        for token in tokenize(phrase):
            if token not in STOPWORDS:
                weighted_terms[token] += max(0.8, weight - rank * 0.12)

    for sample in samples:
        for token in content_tokens(str(sample["en"])):
            weighted_terms[token] += 0.35

    scored: list[tuple[float, str]] = []
    for label, hints in THEME_RULES:
        label_score = 0.0
        for term, weight in weighted_terms.items():
            if term in hints:
                label_score += weight * (1.28 if " " in term else 1.0)
                continue

            token_overlap = len(set(term.split()).intersection(hints))
            if token_overlap:
                label_score += weight * token_overlap * 0.45

        if label_score > 0:
            scored.append((label_score, label))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def infer_cluster_label(
    cluster_id: int,
    phrase_scores: list[tuple[float, str, int, int]],
    samples: list[dict[str, object]],
    video_count: int,
    used_labels: set[str],
) -> tuple[str, str, float, list[tuple[float, str]]]:
    scored_labels = score_theme_labels(phrase_scores, samples)
    best_score = scored_labels[0][0] if scored_labels else 0.0
    best_label = scored_labels[0][1] if scored_labels else ""
    runner_up = scored_labels[1][0] if len(scored_labels) > 1 else 0.0
    margin = best_score - runner_up

    hints = THEME_HINTS_BY_LABEL.get(best_label, set())
    top_phrases = [phrase for _, phrase, _, _ in phrase_scores[:6]]
    phrase_support = count_theme_support(top_phrases, hints)
    sample_support = 0
    for sample in samples:
        sample_phrases = extract_keyphrases(str(sample["en"]))
        if count_theme_support(sample_phrases, hints) > 0:
            sample_support += 1

    raw_confidence = min(1.0, best_score / 13.5) * 0.7 + min(1.0, max(0.0, margin) / 6.0) * 0.3
    if (
        best_label
        and best_score >= THEME_SCORE_THRESHOLD
        and margin >= THEME_MARGIN_THRESHOLD
        and phrase_support >= 2
        and sample_support >= 1
        and video_count >= HIGH_CONFIDENCE_VIDEO_MIN
    ):
        return (
            make_unique_label(best_label, used_labels),
            "theme",
            round(max(0.72, min(0.99, raw_confidence)), 4),
            scored_labels,
        )

    if (
        best_label
        and best_score >= PROVISIONAL_THEME_SCORE_THRESHOLD
        and margin >= PROVISIONAL_THEME_MARGIN_THRESHOLD
        and phrase_support >= 2
        and video_count >= 2
    ):
        provisional_confidence = round(max(0.48, min(0.78, raw_confidence)), 4)
        return (
            make_unique_label(best_label, used_labels),
            "provisional",
            provisional_confidence,
            scored_labels,
        )

    fallback_label = make_unique_label(build_fallback_label(cluster_id, phrase_scores), used_labels)
    fallback_confidence = round(max(0.22, min(0.46, raw_confidence * 0.55 + 0.18)), 4)
    return fallback_label, "descriptive", fallback_confidence, scored_labels


def select_medoid_index(
    vectors: array,
    vector_dim: int,
    members: list[int],
    candidate_pool: list[int],
    similarity_by_member: dict[int, float],
    metadata: list[dict[str, object]],
) -> int:
    best_index = candidate_pool[0]
    best_average_similarity = -float("inf")
    best_center_similarity = similarity_by_member.get(best_index, -float("inf"))
    best_entry_id = str(metadata[best_index]["entryId"])

    for candidate in candidate_pool[:MEDOID_CANDIDATE_COUNT]:
        total_similarity = 0.0
        for member in members:
            total_similarity += dot_rows(vectors, candidate, member, vector_dim)
        average_similarity = total_similarity / max(1, len(members))
        center_similarity = similarity_by_member.get(candidate, -float("inf"))
        entry_id = str(metadata[candidate]["entryId"])

        if (
            average_similarity > best_average_similarity + 1e-12
            or (
                abs(average_similarity - best_average_similarity) <= 1e-12
                and center_similarity > best_center_similarity + 1e-12
            )
            or (
                abs(average_similarity - best_average_similarity) <= 1e-12
                and abs(center_similarity - best_center_similarity) <= 1e-12
                and entry_id < best_entry_id
            )
        ):
            best_index = candidate
            best_average_similarity = average_similarity
            best_center_similarity = center_similarity
            best_entry_id = entry_id

    return best_index


def select_representative_indices(
    metadata: list[dict[str, object]],
    vectors: array,
    vector_dim: int,
    members: list[int],
    center: list[float],
) -> tuple[int, list[int]]:
    scored_members = [
        (dot_row_to_vector(vectors, member, vector_dim, center), member)
        for member in members
    ]
    scored_members.sort(key=lambda item: (-item[0], str(metadata[item[1]]["entryId"])))
    candidate_pool = [member for _, member in scored_members[:REPRESENTATIVE_POOL_COUNT]]
    similarity_by_member = {member: similarity for similarity, member in scored_members}

    medoid_index = select_medoid_index(
        vectors,
        vector_dim,
        members,
        candidate_pool,
        similarity_by_member,
        metadata,
    )
    selected = [medoid_index]
    used_texts = {str(metadata[medoid_index]["en"])}
    used_videos = {str(metadata[medoid_index]["videoId"])}

    while len(selected) < min(REPRESENTATIVE_SAMPLE_COUNT, len(members)):
        best_index: int | None = None
        best_score = -float("inf")
        best_entry_id = ""

        for candidate in candidate_pool:
            if candidate in selected:
                continue

            text_value = str(metadata[candidate]["en"])
            if text_value in used_texts and len(candidate_pool) > len(selected) + 2:
                continue

            center_similarity = similarity_by_member.get(candidate, -1.0)
            similarity_to_selected = max(
                dot_rows(vectors, candidate, selected_index, vector_dim) for selected_index in selected
            )
            novelty = 1.0 - similarity_to_selected
            video_bonus = 0.06 if str(metadata[candidate]["videoId"]) not in used_videos else 0.0
            score = center_similarity * 0.72 + novelty * 0.28 + video_bonus
            entry_id = str(metadata[candidate]["entryId"])

            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and entry_id < best_entry_id
            ):
                best_index = candidate
                best_score = score
                best_entry_id = entry_id

        if best_index is None:
            break

        selected.append(best_index)
        used_texts.add(str(metadata[best_index]["en"]))
        used_videos.add(str(metadata[best_index]["videoId"]))

    return medoid_index, selected


def truncate_text(text: str, limit: int = 96) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[: limit - 3]}..."


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
    if vector_dim <= 0:
        raise SystemExit("Vector dimension could not be inferred.")

    vectors = array("f")
    mean = [0.0] * vector_dim
    metadata: list[dict[str, object]] = []

    for video_id, seg_index, en, zh, start_ms, end_ms, blob in rows:
        unpacked = struct.unpack(f"<{vector_dim}f", blob)
        normalized_vector = normalize(unpacked)
        vectors.extend(normalized_vector)
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

        for dimension, value in enumerate(normalized_vector):
            mean[dimension] += value

    entry_count = len(metadata)
    if entry_count < CLUSTER_COUNT:
        raise SystemExit(
            f"Semantic landscape expects at least {CLUSTER_COUNT} entries, found {entry_count}."
        )

    mean = [value / entry_count for value in mean]

    def centered_dot(row_index: int, direction: list[float]) -> float:
        total = 0.0
        offset = row_offset(row_index, vector_dim)
        for dimension in range(vector_dim):
            total += (vectors[offset + dimension] - mean[dimension]) * direction[dimension]
        return total

    assignments, cluster_centers = spherical_kmeans(vectors, entry_count, vector_dim, CLUSTER_COUNT)

    principal_components: list[list[float]] = []
    for _ in range(2):
        direction = normalize([random.random() - 0.5 for _ in range(vector_dim)])

        for _ in range(PCA_ITERATIONS):
            updated = [0.0] * vector_dim
            for row_index in range(entry_count):
                score = centered_dot(row_index, direction)
                offset = row_offset(row_index, vector_dim)
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

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(assignments):
        cluster_members[cluster_id].append(index)

    if len(cluster_members) != CLUSTER_COUNT:
        missing = [cluster_id for cluster_id in range(CLUSTER_COUNT) if cluster_id not in cluster_members]
        raise SystemExit(f"One or more embedding clusters ended empty: {missing}")

    cluster_phrase_scores = build_cluster_phrase_scores(metadata, assignments)

    clusters_by_id: dict[int, dict[str, object]] = {}
    review_rows: list[dict[str, object]] = []
    used_labels: set[str] = set()
    validation_errors: list[str] = []

    for cluster_id in range(CLUSTER_COUNT):
        members = cluster_members[cluster_id]
        center = cluster_centers[cluster_id]
        medoid_index, representative_indexes = select_representative_indices(
            metadata,
            vectors,
            vector_dim,
            members,
            center,
        )

        samples = [
            {
                "entryId": metadata[index]["entryId"],
                "videoId": metadata[index]["videoId"],
                "segIndex": metadata[index]["segIndex"],
                "en": metadata[index]["en"],
                "zh": metadata[index]["zh"],
            }
            for index in representative_indexes
        ]
        representative_entry_ids = [str(metadata[index]["entryId"]) for index in representative_indexes]
        top_phrase_scores = cluster_phrase_scores.get(cluster_id, [])
        top_phrases = [phrase for _, phrase, _, _ in top_phrase_scores[:6]]
        video_ids = {str(metadata[index]["videoId"]) for index in members}
        video_count = len(video_ids)
        label, label_mode, label_confidence, theme_scores = infer_cluster_label(
            cluster_id,
            top_phrase_scores,
            samples,
            video_count,
            used_labels,
        )

        cluster_x = round(sum(scaled_points[index][0] for index in members) / len(members))
        cluster_y = round(sum(scaled_points[index][1] for index in members) / len(members))
        keyword_list = top_phrases[:]

        clusters_by_id[cluster_id] = {
            "id": cluster_id,
            "label": label,
            "labelMode": label_mode,
            "labelConfidence": label_confidence,
            "color": PALETTE[cluster_id % len(PALETTE)],
            "size": len(members),
            "videoCount": video_count,
            "x": cluster_x,
            "y": cluster_y,
            "keywords": keyword_list,
            "topPhrases": top_phrases,
            "medoidEntryId": str(metadata[medoid_index]["entryId"]),
            "representativeEntryIds": representative_entry_ids,
            "samples": samples,
        }

        best_theme_score = theme_scores[0][0] if theme_scores else 0.0
        runner_up_score = theme_scores[1][0] if len(theme_scores) > 1 else 0.0
        review_rows.append(
            {
                "id": cluster_id,
                "label": label,
                "labelMode": label_mode,
                "labelConfidence": label_confidence,
                "size": len(members),
                "videoCount": video_count,
                "topPhrases": top_phrases,
                "samples": samples,
                "bestThemeScore": best_theme_score,
                "runnerUpThemeScore": runner_up_score,
            }
        )

        if label_mode == "theme" and label_confidence >= HIGH_CONFIDENCE_LABEL_THRESHOLD:
            hints = THEME_HINTS_BY_LABEL.get(label, set())
            phrase_support = count_theme_support(top_phrases, hints)
            sample_support = sum(
                1
                for sample in samples
                if count_theme_support(extract_keyphrases(str(sample["en"])), hints) > 0
            )
            if video_count < HIGH_CONFIDENCE_VIDEO_MIN or phrase_support < 2 or sample_support < 1:
                validation_errors.append(
                    (
                        f"Cluster {cluster_id} is labeled '{label}' at confidence {label_confidence:.2f}, "
                        f"but support is weak (videos={video_count}, phrase_support={phrase_support}, "
                        f"sample_support={sample_support})."
                    )
                )

    print("Semantic landscape review summary")
    for row in sorted(review_rows, key=lambda item: (-int(item["size"]), int(item["id"]))):
        label_mode = str(row["labelMode"])
        confidence = float(row["labelConfidence"])
        print(
            f"- Cluster {row['id']}: {row['label']} [{label_mode}] "
            f"conf={confidence:.2f} entries={row['size']} videos={row['videoCount']}"
        )
        phrases = row["topPhrases"]
        if phrases:
            print(f"  top phrases: {', '.join(str(phrase) for phrase in phrases)}")
        else:
            print("  top phrases: (none)")
        print(
            "  representatives: "
            + " | ".join(
                f"{sample['entryId']}: {truncate_text(str(sample['en']))}"
                for sample in row["samples"]
            )
        )
        print(
            "  theme scores: "
            f"best={float(row['bestThemeScore']):.2f} "
            f"runner-up={float(row['runnerUpThemeScore']):.2f}"
        )

    if validation_errors:
        raise SystemExit("Cluster label validation failed:\n" + "\n".join(validation_errors))

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
        "version": 4,
        "projection": "pca-2d",
        "clusterAlgorithm": "spherical-kmeans-embedding",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceDb": DB_PATH.name,
        "modelId": MODEL_ID,
        "pointCount": entry_count,
        "vectorDim": vector_dim,
        "clusters": clusters,
        "points": points,
        "videoFingerprintWall": build_video_fingerprint_wall(metadata, assignments, scaled_points),
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH} with {entry_count} points.")


if __name__ == "__main__":
    main()
