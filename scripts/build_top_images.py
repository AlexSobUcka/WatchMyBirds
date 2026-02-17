import argparse
import os
import shutil
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_config
from logging_config import get_logger
from utils.collage import (
    build_candidate,
    has_black_border,
    load_image_rgb,
    parse_timestamp,
    trim_dark_borders,
)
from utils.db import get_connection

logger = get_logger(__name__)


def _normalize_species(value: str) -> str:
    """Normalisiert die Artbezeichnung fuer Vergleiche."""
    return str(value or "").strip().lower().replace(" ", "_")


def _fetch_species_rows(conn, species_key: str) -> List[Dict]:
    """Liest alle DB-Zeilen fuer eine Art."""
    cur = conn.execute(
        """
        SELECT
            timestamp,
            original_name,
            optimized_name,
            zoomed_name,
            best_class,
            best_class_conf,
            top1_class_name,
            top1_confidence
        FROM images
        ORDER BY timestamp DESC;
        """
    )
    target = _normalize_species(species_key)
    rows = []
    for row in cur.fetchall():
        species = row["top1_class_name"] or row["best_class"]
        if _normalize_species(species) != target:
            continue
        rows.append(row)
    return rows


def _fetch_species_list(conn) -> List[str]:
    """Liest alle vorhandenen Arten aus der Datenbank."""
    cur = conn.execute(
        """
        SELECT DISTINCT COALESCE(NULLIF(top1_class_name, ''), NULLIF(best_class, '')) AS species
        FROM images
        WHERE COALESCE(NULLIF(top1_class_name, ''), NULLIF(best_class, '')) IS NOT NULL
        ORDER BY species;
        """
    )
    return [row["species"] for row in cur.fetchall() if row["species"]]


def _select_image_path(row, output_dir: str, prefer_zoomed: bool, avoid_black: bool):
    """Waehlt den Bildpfad fuer eine DB-Zeile aus."""
    timestamp_value = row["timestamp"] or ""
    date_prefix = timestamp_value.split("_")[0]
    zoomed_name = row["zoomed_name"] or ""
    optimized_name = row["optimized_name"] or ""
    original_name = row["original_name"] or ""

    zoomed_path = (
        os.path.join(output_dir, date_prefix, zoomed_name) if zoomed_name else ""
    )
    optimized_path = (
        os.path.join(output_dir, date_prefix, optimized_name) if optimized_name else ""
    )
    original_path = (
        os.path.join(output_dir, date_prefix, original_name) if original_name else ""
    )

    if prefer_zoomed and zoomed_path and os.path.exists(zoomed_path):
        image = load_image_rgb(zoomed_path)
        if image is not None and (not avoid_black or not has_black_border(image)):
            return zoomed_path, image
    if optimized_path and os.path.exists(optimized_path):
        image = load_image_rgb(optimized_path)
        if image is not None:
            return optimized_path, image
    if original_path and os.path.exists(original_path):
        image = load_image_rgb(original_path)
        if image is not None:
            return original_path, image
    return "", None


def _build_candidates(rows, output_dir: str, config: Dict) -> List[Dict]:
    """Erstellt Kandidaten mit Metriken fuer das Ranking."""
    prefer_zoomed = bool(config.get("COLLAGE_USE_ZOOMED", True))
    avoid_black = bool(config.get("COLLAGE_AVOID_BLACK_BORDERS", True))
    candidates = []
    for row in rows:
        image_path, image = _select_image_path(
            row, output_dir, prefer_zoomed=prefer_zoomed, avoid_black=avoid_black
        )
        if not image_path or image is None:
            continue
        if avoid_black:
            image = trim_dark_borders(image)
        classifier_conf = row["top1_confidence"]
        detector_conf = row["best_class_conf"]
        try:
            classifier_conf = float(classifier_conf) if classifier_conf is not None else 0.0
        except Exception:
            classifier_conf = 0.0
        try:
            detector_conf = float(detector_conf) if detector_conf is not None else 0.0
        except Exception:
            detector_conf = 0.0
        timestamp_value = parse_timestamp(row["timestamp"] or "")
        candidate = build_candidate(
            image_path,
            classifier_conf if classifier_conf > 0 else detector_conf,
            timestamp_value,
            config,
            image=image,
        )
        if candidate is None:
            continue
        candidate["classifier_confidence"] = classifier_conf
        candidate["detector_confidence"] = detector_conf
        candidate["timestamp_raw"] = row["timestamp"] or ""
        candidates.append(candidate)
    return candidates


def _score_candidate(candidate: Dict, mode: str) -> float:
    """Berechnet den Score fuer einen Modus."""
    if mode == "brightness":
        return float(candidate.get("brightness", 0.0))
    if mode == "classifier_confidence":
        return float(candidate.get("classifier_confidence", 0.0))
    if mode == "detector_confidence":
        return float(candidate.get("detector_confidence", 0.0))
    if mode == "sharpness":
        return float(candidate.get("sharpness", 0.0))
    if mode == "combo_score":
        return float(candidate.get("score", 0.0))
    return 0.0


def _save_top_images(
    candidates: List[Dict],
    output_dir: str,
    species_key: str,
    mode: str,
    limit: int,
) -> List[str]:
    """Speichert die Top-N Bilder in ein Zielverzeichnis."""
    safe_species = _normalize_species(species_key) or "unknown"
    target_dir = os.path.join(output_dir, "telegram", "top_images", safe_species, mode)
    os.makedirs(target_dir, exist_ok=True)

    scored = sorted(
        candidates, key=lambda item: _score_candidate(item, mode), reverse=True
    )
    saved = []
    for idx, item in enumerate(scored[:limit], start=1):
        src = item.get("path", "")
        if not src or not os.path.exists(src):
            continue
        timestamp_text = item.get("timestamp_raw", "unknown")
        ext = os.path.splitext(src)[1] or ".jpg"
        filename = f"{idx:02d}_{timestamp_text}{ext}"
        dest = os.path.join(target_dir, filename)
        try:
            shutil.copy2(src, dest)
            saved.append(dest)
        except Exception as exc:
            logger.warning("Failed to copy '%s' to '%s': %s", src, dest, exc)
    return saved


def _parse_modes(mode_text: str, allowed_modes: List[str]) -> List[str]:
    """Parst eine Modusliste, die kommasepariert uebergeben wird."""
    if not mode_text:
        return []
    if mode_text.strip().lower() == "all":
        return allowed_modes
    parts = [part.strip() for part in mode_text.split(",")]
    return [part for part in parts if part]


def main():
    """Erstellt Top-N Bilder fuer eine oder mehrere Arten nach Modi."""
    parser = argparse.ArgumentParser(
        description=(
            "Build top-N images for a species based on brightness, confidence, or "
            "combined scores."
        )
    )
    parser.add_argument("--species", help="Species name to filter.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all species found in the database.",
    )
    parser.add_argument(
        "--mode",
        default="brightness",
        choices=[
            "brightness",
            "classifier_confidence",
            "detector_confidence",
            "sharpness",
            "combo_score",
        ],
        help="Ranking mode.",
    )
    parser.add_argument(
        "--modes",
        default="",
        help="Comma-separated list of modes (or 'all') to run.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of images.")
    args = parser.parse_args()

    config = get_config()
    conn = get_connection()
    output_dir = config["OUTPUT_DIR"]

    allowed_modes = [
        "brightness",
        "classifier_confidence",
        "detector_confidence",
        "sharpness",
        "combo_score",
    ]
    selected_modes = _parse_modes(args.modes, allowed_modes)
    if not selected_modes:
        selected_modes = [args.mode]

    if args.all:
        species_list = _fetch_species_list(conn)
        if not species_list:
            print("No species found in the database.")
            return
    else:
        if not args.species:
            print("Provide --species or use --all.")
            return
        species_list = [args.species]

    for species in species_list:
        rows = _fetch_species_rows(conn, species)
        if not rows:
            print(f"No images found for species: {species}")
            continue

        candidates = _build_candidates(rows, output_dir, config)
        if not candidates:
            print(f"No usable images found for species: {species}")
            continue

        for mode in selected_modes:
            saved = _save_top_images(
                candidates, output_dir, species, mode, args.limit
            )
            if not saved:
                print(f"No images saved for species: {species} (mode={mode})")
                continue
            print(f"Saved {len(saved)} images to: {os.path.dirname(saved[0])}")


if __name__ == "__main__":
    main()
