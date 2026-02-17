from datetime import datetime

from config import get_config
from logging_config import get_logger
from utils.collage import (
    build_candidate,
    build_square_tile,
    has_black_border,
    load_image_rgb,
    parse_timestamp,
    select_collage_candidates,
    trim_dark_borders,
)
from utils.db import fetch_daily_species_summary, fetch_day_images, get_connection

import os
from PIL import Image

logger = get_logger(__name__)


def _fetch_recent_dates(conn, limit):
    """Liest die letzten Datums-Keys aus der Datenbank."""
    cur = conn.execute(
        """
        SELECT DISTINCT substr(timestamp, 1, 8) AS date_key
        FROM images
        ORDER BY date_key DESC
        LIMIT ?;
        """,
        (limit,),
    )
    return [row["date_key"] for row in cur.fetchall() if row["date_key"]]


def _collect_candidates(rows, output_dir, config):
    """Sammelt Kandidaten pro Art mit Metriken fuer die Collage."""
    images_by_species = {}
    for row in rows:
        species = row["top1_class_name"] or row["best_class"]
        if not species:
            continue
        timestamp_value = parse_timestamp(row["timestamp"] or "")
        date_prefix = (row["timestamp"] or "").split("_")[0]
        zoomed_name = row["zoomed_name"] or ""
        optimized_name = row["optimized_name"] or ""
        zoomed_path = (
            os.path.join(output_dir, date_prefix, zoomed_name) if zoomed_name else ""
        )
        optimized_path = (
            os.path.join(output_dir, date_prefix, optimized_name)
            if optimized_name
            else ""
        )
        prefer_zoomed = bool(config.get("COLLAGE_USE_ZOOMED", True))
        avoid_black = bool(config.get("COLLAGE_AVOID_BLACK_BORDERS", True))
        selected_path = ""
        selected_image = None
        if prefer_zoomed and zoomed_path and os.path.exists(zoomed_path):
            selected_image = load_image_rgb(zoomed_path)
            if selected_image is None:
                selected_path = ""
            elif avoid_black and has_black_border(selected_image):
                selected_path = ""
            else:
                selected_path = zoomed_path
        if not selected_path and optimized_path and os.path.exists(optimized_path):
            selected_path = optimized_path
            selected_image = load_image_rgb(optimized_path)
        if not selected_path or selected_image is None:
            continue
        if avoid_black:
            selected_image = trim_dark_borders(selected_image)
        confidence = row["top1_confidence"]
        try:
            confidence = float(confidence) if confidence is not None else 0.0
        except Exception:
            confidence = 0.0
        candidate = build_candidate(
            selected_path,
            confidence,
            timestamp_value,
            config,
            image=selected_image,
        )
        if candidate is None:
            continue
        images_by_species.setdefault(species, []).append(candidate)
    return images_by_species


def _build_collage(image_paths, output_path, avoid_black, cell_size=320):
    """Erstellt eine quadratische Collage-Datei."""
    if not image_paths:
        return None
    grid_map = {1: 1, 4: 2, 9: 3}
    grid = grid_map.get(len(image_paths))
    if not grid:
        return None
    collage_size = (grid * cell_size, grid * cell_size)
    collage = Image.new("RGB", collage_size, color=(0, 0, 0))

    for idx, image_path in enumerate(image_paths):
        row = idx // grid
        col = idx % grid
        image = load_image_rgb(image_path)
        if image is None:
            continue
        if avoid_black:
            image = trim_dark_borders(image)
        tile = build_square_tile(image, cell_size)
        collage.paste(tile, (col * cell_size, row * cell_size))

    try:
        collage.save(output_path, "JPEG", quality=85)
    except Exception as exc:
        logger.warning("Failed to save collage '%s': %s", output_path, exc)
        return None
    return output_path


def main():
    """Baut Collagen fuer die letzten drei Tage fuer Debug-Zwecke."""
    config = get_config()
    conn = get_connection()
    output_dir = config["OUTPUT_DIR"]
    recent_dates = _fetch_recent_dates(conn, limit=3)
    if not recent_dates:
        print("No images found in the database.")
        return

    collage_dir = os.path.join(output_dir, "telegram", "debug")
    os.makedirs(collage_dir, exist_ok=True)

    for date_key in recent_dates:
        date_str = datetime.strptime(date_key, "%Y%m%d").strftime("%Y-%m-%d")
        rows = fetch_day_images(conn, date_str)
        if not rows:
            continue
        images_by_species = _collect_candidates(rows, output_dir, config)
        if not images_by_species:
            continue
        summary_rows = fetch_daily_species_summary(conn, date_str)
        species_order = [
            row["species"]
            for row in summary_rows
            if row["species"] in images_by_species
        ]
        for species in species_order:
            items = images_by_species.get(species, [])
            selected = select_collage_candidates(items, config)
            if not selected:
                continue
            selected_paths = [item["path"] for item in selected]
            grid_size = len(selected_paths)
            safe_species = "".join(
                c if c.isalnum() or c in ("_", "-") else "_" for c in str(species)
            )
            collage_name = f"{date_str}_{safe_species}_{grid_size}.jpg"
            collage_path = os.path.join(collage_dir, collage_name)
            result_path = _build_collage(
                selected_paths,
                collage_path,
                avoid_black=bool(config.get("COLLAGE_AVOID_BLACK_BORDERS", True)),
            )
            if result_path:
                print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
