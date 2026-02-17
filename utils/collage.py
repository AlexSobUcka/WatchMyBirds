from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from logging_config import get_logger

logger = get_logger(__name__)

def parse_timestamp(timestamp_text: str) -> Optional[datetime]:
    """Parst einen Bild-Timestamp im Format YYYYMMDD_HHMMSS."""
    if not timestamp_text:
        return None
    try:
        return datetime.strptime(timestamp_text, "%Y%m%d_%H%M%S")
    except Exception:
        return None


def load_image_rgb(image_path: str) -> Optional[Image.Image]:
    """Laedt ein Bild als RGB und gibt None bei Fehlern zurueck."""
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def compute_brightness(image: Image.Image) -> float:
    """Berechnet die mittlere Bildhelligkeit im Bereich 0..1."""
    gray = image.convert("L")
    brightness = float(np.mean(np.array(gray))) / 255.0
    return max(0.0, min(1.0, brightness))


def compute_sharpness(image: Image.Image) -> float:
    """Berechnet eine Schaerfe-Metrik via Laplacian-Varianz."""
    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Erstellt einen dHash fuer Aehnlichkeitsvergleiche."""
    resample = getattr(Image, "Resampling", Image).LANCZOS
    resized = image.convert("L").resize((hash_size + 1, hash_size), resample)
    pixels = np.array(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_value = 0
    for bit in diff.flatten():
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Zaehlt die unterschiedlichen Bits zwischen zwei Hashes."""
    return int(bin(hash_a ^ hash_b).count("1"))


def has_black_border(
    image: Image.Image, threshold: float = 0.05, border_ratio: float = 0.08
) -> bool:
    """Prueft, ob dunkle Rander den Randbereich dominieren."""
    gray = np.array(image.convert("L"))
    height, width = gray.shape[:2]
    border = max(1, int(min(width, height) * border_ratio))
    top = gray[:border, :]
    bottom = gray[-border:, :]
    left = gray[:, :border]
    right = gray[:, -border:]
    border_pixels = np.concatenate(
        (top.flatten(), bottom.flatten(), left.flatten(), right.flatten())
    )
    border_mean = float(np.mean(border_pixels)) / 255.0
    return border_mean < threshold


def trim_dark_borders(
    image: Image.Image, threshold: float = 0.05, max_trim_ratio: float = 0.2
) -> Image.Image:
    """Schneidet dunkle Rander bis zu einer Maximalbreite ab."""
    gray = np.array(image.convert("L"))
    height, width = gray.shape[:2]
    max_trim = max(1, int(min(width, height) * max_trim_ratio))

    def is_dark_row(row_idx: int) -> bool:
        return (float(np.mean(gray[row_idx, :])) / 255.0) < threshold

    def is_dark_col(col_idx: int) -> bool:
        return (float(np.mean(gray[:, col_idx])) / 255.0) < threshold

    top = 0
    while top < max_trim and is_dark_row(top):
        top += 1
    bottom = height - 1
    while height - 1 - bottom < max_trim and is_dark_row(bottom):
        bottom -= 1
    left = 0
    while left < max_trim and is_dark_col(left):
        left += 1
    right = width - 1
    while width - 1 - right < max_trim and is_dark_col(right):
        right -= 1

    if left >= right or top >= bottom:
        return image
    if top == 0 and bottom == height - 1 and left == 0 and right == width - 1:
        return image
    return image.crop((left, top, right + 1, bottom + 1))


def build_square_tile(image: Image.Image, size_px: int) -> Image.Image:
    """Erstellt ein quadratisches Tile per zentriertem Zuschnitt."""
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return ImageOps.fit(image, (size_px, size_px), method=resample, centering=(0.5, 0.5))


def score_candidate(candidate: Dict, config: Dict) -> float:
    """Berechnet den Score fuer die Collage-Auswahl."""
    confidence_weight = float(config.get("COLLAGE_CONFIDENCE_WEIGHT", 1.0))
    brightness_weight = float(config.get("COLLAGE_BRIGHTNESS_WEIGHT", 0.3))
    sharpness_weight = float(config.get("COLLAGE_SHARPNESS_WEIGHT", 0.2))
    sharpness_target = float(config.get("COLLAGE_SHARPNESS_TARGET", 120.0))
    sharpness = float(candidate.get("sharpness", 0.0))
    if sharpness_target > 0:
        sharpness_score = min(sharpness / sharpness_target, 1.0)
    else:
        sharpness_score = 0.0
    return (
        confidence_weight * float(candidate.get("confidence", 0.0))
        + brightness_weight * float(candidate.get("brightness", 0.0))
        + sharpness_weight * sharpness_score
    )


def build_candidate(
    image_path: str,
    confidence: float,
    timestamp_value: Optional[datetime],
    config: Dict,
    image: Optional[Image.Image] = None,
) -> Optional[Dict]:
    """Erstellt einen Collage-Kandidaten mit Metriken."""
    if image is None:
        image = load_image_rgb(image_path)
    if image is None:
        return None
    brightness = compute_brightness(image)
    sharpness = compute_sharpness(image)
    dhash_value = compute_dhash(image)
    border_threshold = float(config.get("COLLAGE_BLACK_BORDER_THRESHOLD", 0.05))
    border_ratio = float(config.get("COLLAGE_BLACK_BORDER_RATIO", 0.08))
    has_border = has_black_border(
        image, threshold=border_threshold, border_ratio=border_ratio
    )
    candidate = {
        "path": image_path,
        "confidence": float(confidence),
        "timestamp": timestamp_value,
        "brightness": brightness,
        "sharpness": sharpness,
        "dhash": dhash_value,
        "has_black_border": has_border,
    }
    candidate["score"] = score_candidate(candidate, config)
    return candidate


def select_collage_candidates(candidates: list, config: Dict) -> list:
    """Waehlt die besten Kandidaten mit Filter und Diversitaet."""
    if not candidates:
        return []
    max_images = int(config.get("COLLAGE_MAX_IMAGES", 9))
    grid_sizes = [9, 4, 1]
    grid_sizes = [size for size in grid_sizes if size <= max_images]
    target_count = 1
    for size in grid_sizes:
        if len(candidates) >= size:
            target_count = size
            break

    min_conf = float(config.get("COLLAGE_MIN_CONFIDENCE", 0.0))
    min_brightness = float(config.get("COLLAGE_MIN_BRIGHTNESS", 0.0))
    min_sharpness = float(config.get("COLLAGE_MIN_SHARPNESS", 0.0))
    min_gap_minutes = float(config.get("COLLAGE_MIN_TIME_GAP_MINUTES", 0.0))
    dhash_threshold = int(config.get("COLLAGE_DHASH_THRESHOLD", 0))
    avoid_black = bool(config.get("COLLAGE_AVOID_BLACK_BORDERS", True))
    debug_enabled = bool(config.get("COLLAGE_DEBUG", False))

    filtered = []
    for item in candidates:
        if item.get("confidence", 0.0) < min_conf:
            continue
        if item.get("brightness", 0.0) < min_brightness:
            continue
        if item.get("sharpness", 0.0) < min_sharpness:
            continue
        if avoid_black and item.get("has_black_border"):
            continue
        filtered.append(item)
    if not filtered:
        if debug_enabled:
            logger.info(
                "Collage select: total=%d filtered=0 (min filters/black borders).",
                len(candidates),
            )
        return []

    bucketed = filtered
    if min_gap_minutes > 0:
        buckets = {}
        for idx, item in enumerate(filtered):
            timestamp = item.get("timestamp")
            if timestamp is None:
                bucket_key = ("none", idx)
            else:
                bucket_key = int(timestamp.timestamp() // (min_gap_minutes * 60))
            existing = buckets.get(bucket_key)
            if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
                buckets[bucket_key] = item
        bucketed = list(buckets.values())

    grouped = bucketed
    if dhash_threshold > 0:
        prefix_bits = max(1, 64 - dhash_threshold * 2)
        prefix_bits = min(64, prefix_bits)
        if prefix_bits == 64:
            mask = (1 << 64) - 1
        else:
            mask = ((1 << prefix_bits) - 1) << (64 - prefix_bits)
        groups = {}
        for idx, item in enumerate(bucketed):
            dhash_value = item.get("dhash")
            if dhash_value is None:
                group_key = ("none", idx)
            else:
                group_key = dhash_value & mask
            existing = groups.get(group_key)
            if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
                groups[group_key] = item
        grouped = list(groups.values())

    scored = sorted(grouped, key=lambda x: x.get("score", 0.0), reverse=True)
    if not scored:
        if debug_enabled:
            logger.info(
                "Collage select: total=%d filtered=%d bucketed=%d grouped=0.",
                len(candidates),
                len(filtered),
                len(bucketed),
            )
        return []

    target_count = 1
    for size in grid_sizes:
        if len(scored) >= size:
            target_count = size
            break
    selected = scored[:target_count]
    if debug_enabled:
        logger.info(
            "Collage select: total=%d filtered=%d bucketed=%d grouped=%d target=%d selected=%d.",
            len(candidates),
            len(filtered),
            len(bucketed),
            len(grouped),
            target_count,
            len(selected),
        )
    return selected
