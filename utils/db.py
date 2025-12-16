import os
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, List

from config import get_config


DB_FILENAME = "images.db"


def _get_db_path() -> Path:
    cfg = get_config()
    output_dir = Path(cfg.get("OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / DB_FILENAME


def get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    _init_schema(conn)
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            timestamp TEXT PRIMARY KEY,
            original_name TEXT,
            optimized_name TEXT,
            zoomed_name TEXT,
            best_class TEXT,
            best_class_conf REAL,
            top1_class_name TEXT,
            top1_confidence REAL,
            coco_json TEXT,
            downloaded_timestamp TEXT
        );
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(timestamp DESC);"
    )


def insert_image(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO images (
            timestamp,
            original_name,
            optimized_name,
            zoomed_name,
            best_class,
            best_class_conf,
            top1_class_name,
            top1_confidence,
            coco_json,
            downloaded_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("timestamp"),
            row.get("original_name"),
            row.get("optimized_name"),
            row.get("zoomed_name"),
            row.get("best_class"),
            row.get("best_class_conf"),
            row.get("top1_class_name"),
            row.get("top1_confidence"),
            row.get("coco_json"),
            row.get("downloaded_timestamp", ""),
        ),
    )
    conn.commit()


def fetch_all_images(conn: sqlite3.Connection) -> List[sqlite3.Row]:
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
            top1_confidence,
            coco_json,
            downloaded_timestamp
        FROM images
        ORDER BY timestamp DESC;
        """
    )
    return cur.fetchall()


def fetch_images_by_date(conn: sqlite3.Connection, date_str_iso: str) -> List[sqlite3.Row]:
    """date_str_iso: YYYY-MM-DD; matches timestamp prefix YYYYMMDD_."""
    date_prefix = date_str_iso.replace("-", "")
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
            top1_confidence,
            coco_json,
            downloaded_timestamp
        FROM images
        WHERE timestamp LIKE ? || '%'
        ORDER BY timestamp DESC;
        """,
        (date_prefix,),
    )
    return cur.fetchall()


def delete_images_by_names(conn: sqlite3.Connection, optimized_names: Iterable[str]) -> None:
    names = list(optimized_names)
    if not names:
        return
    placeholders = ",".join("?" for _ in names)
    conn.execute(
        f"DELETE FROM images WHERE optimized_name IN ({placeholders});", names
    )
    conn.commit()


def update_downloaded_timestamp(
    conn: sqlite3.Connection, optimized_names: Iterable[str], download_ts: str
) -> None:
    names = list(optimized_names)
    if not names:
        return
    placeholders = ",".join("?" for _ in names)
    params = [download_ts] + names
    conn.execute(
        f"""
        UPDATE images
        SET downloaded_timestamp = ?
        WHERE optimized_name IN ({placeholders});
        """,
        params,
    )
    conn.commit()
