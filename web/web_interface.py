# ------------------------------------------------------------------------------
# web_interface.py
# ------------------------------------------------------------------------------

import os
import json
import math
from urllib.parse import parse_qs
import re
import logging
from flask import Flask, send_from_directory, Response, request, jsonify
from dash import (
    Dash,
    html,
    dcc,
    callback_context,
    ALL,
    Input,
    Output,
    State,
    no_update,
    ClientsideFunction,
)
import dash_bootstrap_components as dbc
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
import plotly.express as px
from config import (
    get_config,
    get_settings_payload,
    validate_runtime_updates,
    update_runtime_settings,
)
from utils.db import (
    get_connection,
    fetch_all_images,
    fetch_daily_covers,
    fetch_images_by_date,
    delete_images_by_names,
    update_downloaded_timestamp,
    fetch_image_summaries,
    fetch_day_images,
    fetch_hourly_counts,
    fetch_day_count,
    fetch_daily_species_summary,
)

config = get_config()
db_conn = get_connection()

import zipfile
import io  # Create in-memory zip buffer
import base64  # Send zip data to dcc.Download
from dash.exceptions import PreventUpdate
import pandas as pd


# >>> Caching settings for gallery functions >>>
_CACHE_TIMEOUT = 60  # Set cache timeout in seconds
_cached_images = {"images": None, "timestamp": 0}
_species_summary_cache = {"payload": None, "timestamp": 0}
_daily_species_summary_cache = {}
_DAILY_SPECIES_CACHE_TTL = 300  # seconds
_daily_gallery_summary_cache = {}
_DAILY_GALLERY_SUMMARY_TTL = 300  # seconds


def create_web_interface(detection_manager):
    """
    Creates and returns a web interface (Dash app and Flask server) for the project.
    Expects the following keys in params:
      - output_dir: Directory where detection images are saved.
      - output_resize_width: The width to which video frames should be resized.
      - STREAM_FPS: Frame rate for the video stream.
      - IMAGE_WIDTH: Width for thumbnail images.
      - RECENT_IMAGES_COUNT: Number of recent images to display.
      - PAGE_SIZE: Number of images per gallery page.
    """
    logger = logging.getLogger(__name__)

    output_dir = config["OUTPUT_DIR"]
    output_resize_width = config["STREAM_WIDTH_OUTPUT_RESIZE"]
    CONFIDENCE_THRESHOLD_DETECTION = config["CONFIDENCE_THRESHOLD_DETECTION"]
    CLASSIFIER_CONFIDENCE_THRESHOLD = config["CLASSIFIER_CONFIDENCE_THRESHOLD"]
    FUSION_ALPHA = config["FUSION_ALPHA"]
    EDIT_PASSWORD = config["EDIT_PASSWORD"]
    logger.info(
        f"Loaded EDIT_PASSWORD: {'***' if EDIT_PASSWORD and EDIT_PASSWORD != 'default_pass' else '<Not Set or Default>'}"
    )

    if EDIT_PASSWORD == "default_pass":
        logger.warning(
            "EDIT_PASSWORD not set in .env file, using default. THIS IS INSECURE."
        )

    RECENT_IMAGES_COUNT = 10
    IMAGE_WIDTH = 150
    PAGE_SIZE = 50

    common_names_file = os.path.join(os.getcwd(), "assets", "common_names_DE.json")
    try:
        with open(common_names_file, "r", encoding="utf-8") as f:
            COMMON_NAMES = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load common names from {common_names_file}: {e}")
        COMMON_NAMES = {"Cyanistes_caeruleus": "Eurasian blue tit"}

    # >>> Helper Functions for DB and File Operations >>>
    def _rows_to_df(rows):
        if not rows:
            return pd.DataFrame()
        records = [dict(row) for row in rows]
        df = pd.DataFrame.from_records(records)
        if "downloaded_timestamp" not in df.columns:
            df["downloaded_timestamp"] = ""
        return df.sort_values(by="timestamp", ascending=False)

    def read_csv_for_date(date_str_iso):  # legacy name; now reads from SQLite
        rows = fetch_images_by_date(db_conn, date_str_iso)
        return _rows_to_df(rows)

    def delete_image_files(relative_optimized_path):
        """Deletes original, optimized, and zoomed versions of an image."""
        base_path = os.path.join(output_dir, relative_optimized_path)
        original_path = base_path.replace("_optimized", "_original")
        zoomed_path = derive_zoomed_filename(base_path)  # Use existing helper

        deleted_count = 0
        for img_path in [original_path, base_path, zoomed_path]:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    logger.info(f"Deleted image file: {img_path}")
                    deleted_count += 1
            except OSError as e:
                logger.error(f"Error deleting file {img_path}: {e}")
        return deleted_count > 0  # Return True if at least one file was deleted

    def get_all_images():
        """
        Reads all rows from SQLite and returns a list of tuples:
          (timestamp, optimized image relative path, best_class, best_class_conf, top1_class_name, top1_confidence)
        Sorted by timestamp (newest first).
        """
        images = []
        try:
            rows = fetch_all_images(db_conn)
            for row in rows:
                timestamp = row["timestamp"]
                optimized_name = row["optimized_name"]
                best_class = row["best_class"]
                best_class_conf = row["best_class_conf"]
                top1_class = row["top1_class_name"]
                top1_conf = row["top1_confidence"]
                if not timestamp or not optimized_name:
                    continue
                date_folder = timestamp[:8]
                rel_path = os.path.join(date_folder, optimized_name)
                images.append(
                    (
                        timestamp,
                        rel_path,
                        best_class,
                        best_class_conf,
                        top1_class,
                        top1_conf,
                    )
                )
        except Exception as e:
            logger.error(f"Error reading images from SQLite: {e}")
        return images

    def get_daily_covers():
        """Returns a dict of {YYYY-MM-DD: relative_path} for gallery overview."""
        covers = {}
        try:
            rows = fetch_daily_covers(db_conn)
            for row in rows:
                date_key = row["date_key"]
                optimized_name = row["optimized_name"]
                if not date_key or not optimized_name:
                    continue
                formatted_date = (
                    f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:]}"
                )
                rel_path = os.path.join(date_key, optimized_name)
                covers[formatted_date] = rel_path
        except Exception as e:
            logger.error(f"Error reading daily covers from SQLite: {e}")
        return covers

    def get_captured_images():
        """
        Returns a list of captured optimized images without coco_json.
        Uses caching to avoid repeated disk reads.
        """
        now = time.time()
        if (
            _cached_images["images"] is not None
            and (now - _cached_images["timestamp"]) < _CACHE_TIMEOUT
        ):
            return _cached_images["images"]
        images = []
        try:
            rows = fetch_image_summaries(db_conn)
            for row in rows:
                timestamp = row["timestamp"]
                optimized_name = row["optimized_name"]
                best_class = row["best_class"]
                best_class_conf = row["best_class_conf"]
                top1_class = row["top1_class_name"]
                top1_conf = row["top1_confidence"]
                if not timestamp or not optimized_name:
                    continue
                date_folder = timestamp[:8]
                rel_path = os.path.join(date_folder, optimized_name)
                images.append(
                    (
                        timestamp,
                        rel_path,
                        best_class,
                        best_class_conf,
                        top1_class,
                        top1_conf,
                    )
                )
        except Exception as e:
            logger.error(f"Error reading image summaries from SQLite: {e}")
        _cached_images["images"] = images
        _cached_images["timestamp"] = now
        return images

    def get_captured_images_by_date():
        """
        Returns a dictionary grouping images by date (YYYY-MM-DD) using the CSV-based image list.
        The grouping is done by extracting the date from the optimized filename (which is expected to start with YYYYMMDD).
        """
        images = (
            get_captured_images()
        )  # now each element is a tuple: (timestamp, filename, best_class, best_class_conf, top1_class_name, top1_confidence)
        images_by_date = {}
        for (
            timestamp,
            filename,
            best_class,
            best_class_conf,
            top1_class,
            top1_conf,
        ) in images:
            base = os.path.basename(filename)
            match = re.match(r"(\d{8})_\d{6}.*\.jpg", base)
            if match:
                date_str = match.group(1)  # Extract YYYYMMDD
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                if formatted_date not in images_by_date:
                    images_by_date[formatted_date] = []
                images_by_date[formatted_date].append(
                    (filename, best_class, best_class_conf, top1_class, top1_conf)
                )
        return images_by_date

    def get_daily_species_summary(date_iso: str):
        """Returns cached per-species counts for a given date (YYYY-MM-DD)."""
        now = time.time()
        cached = _daily_species_summary_cache.get(date_iso)
        if cached and (now - cached["timestamp"]) < _DAILY_SPECIES_CACHE_TTL:
            return cached["data"]
        try:
            rows = fetch_daily_species_summary(db_conn, date_iso)
        except Exception as e:
            logger.error(f"Error fetching daily species summary for {date_iso}: {e}")
            rows = []

        summary = []
        for row in rows:
            species = row["species"]
            count = row["count"]
            if not species:
                continue
            common_name = COMMON_NAMES.get(species, species.replace("_", " "))
            summary.append(
                {"species": species, "common_name": common_name, "count": int(count)}
            )
        _daily_species_summary_cache[date_iso] = {"data": summary, "timestamp": now}
        return summary

    def get_gallery_daily_summary(date_iso: str):
        """Returns cached gallery-style daily summary content for landing."""
        now = time.time()
        cached = _daily_gallery_summary_cache.get(date_iso)
        if cached and (now - cached["timestamp"]) < _DAILY_GALLERY_SUMMARY_TTL:
            return cached["content"]

        try:
            rows = fetch_day_images(db_conn, date_iso)
        except Exception as e:
            logger.error(f"Error fetching day images for daily summary {date_iso}: {e}")
            rows = []

        images_for_this_date = []
        for row in rows:
            optimized_name = row["optimized_name"]
            if not optimized_name:
                continue
            rel_path = os.path.join(date_iso.replace("-", ""), optimized_name)
            images_for_this_date.append(
                (
                    rel_path,
                    row["best_class"],
                    row["best_class_conf"],
                    row["top1_class_name"],
                    row["top1_confidence"],
                )
            )

        if not images_for_this_date:
            content = html.P(
                f"No detections found for {date_iso}.",
                className="text-center text-muted",
            )
        else:
            agreement_summary_content = generate_daily_fused_summary_agreement(
                date_iso, images_for_this_date
            )
            weighted_summary_content = generate_daily_fused_summary_weighted(
                date_iso, images_for_this_date
            )
            content = html.Div(
                [
                    html.H4(
                        "Daily Summary: Agreement & Product Score",
                        className="text-center mt-4",
                    ),
                    agreement_summary_content,
                    html.H4(
                        f"Daily Summary: Weighted Score, α={FUSION_ALPHA}",
                        className="text-center mt-4",
                    ),
                    weighted_summary_content,
                ]
            )

        _daily_gallery_summary_cache[date_iso] = {"content": content, "timestamp": now}
        return content

    def derive_zoomed_filename(
        optimized_filename: str,
        optimized_suffix: str = "_optimized",
        zoomed_suffix: str = "_zoomed",
    ) -> str:
        """Derives the zoomed image filename from the optimized filename."""
        return optimized_filename.replace(optimized_suffix, zoomed_suffix)

    def build_edit_tiles(date_str_iso: str, df_input: pd.DataFrame):
        """Builds edit tiles for a given date and dataframe."""
        tiles = []
        for _, row in df_input.iterrows():
            relative_path = os.path.join(
                date_str_iso.replace("-", ""), row["optimized_name"]
            )
            checklist_value = relative_path

            info_box = create_thumbnail_info_box(
                row.get("best_class", ""),
                row.get("best_class_conf", ""),
                row.get("top1_class_name", ""),
                row.get("top1_confidence", ""),
            )
            downloaded_ts = row.get("downloaded_timestamp", "")
            if downloaded_ts and str(downloaded_ts).strip():
                info_box.children.append(
                    html.Span(
                        "Downloaded",
                        className="info-download-status text-success small",
                    )
                )

            zoomed_filename = derive_zoomed_filename(relative_path)

            checkbox_component = dbc.Checkbox(
                id={"type": "edit-image-checkbox", "index": checklist_value},
                value=False,
                className="edit-checkbox",
            )

            tile_classname = "gallery-tile edit-tile"
            if downloaded_ts and str(downloaded_ts).strip():
                tile_classname += " downloaded-image"

            tiles.append(
                html.Div(
                    [
                        html.Div(
                            checkbox_component,
                            className="edit-checkbox-wrapper",
                        ),
                        html.Button(
                            [
                                html.Img(
                                    src=f"/images/{zoomed_filename}",
                                    alt=f"Thumbnail {row['optimized_name']}",
                                    className="thumbnail-image",
                                    style={"width": f"{IMAGE_WIDTH}px"},
                                )
                            ],
                            id={"type": "edit-image-toggle", "index": checklist_value},
                            n_clicks=0,
                            className="edit-thumb-button",
                        ),
                        info_box,
                    ],
                    className=tile_classname,
                )
            )
        return tiles

    def create_thumbnail_button(image_filename: str, index: int, id_type: str):
        """Creates a clickable thumbnail button with standard styling."""
        zoomed_filename = derive_zoomed_filename(image_filename)
        return html.Button(
            html.Img(
                src=f"/images/{zoomed_filename}",
                alt=f"Thumbnail of {zoomed_filename}",
                className="thumbnail-image",
                style={"width": f"{IMAGE_WIDTH}px"},
            ),
            id={"type": id_type, "index": index},
            n_clicks=0,
            className="thumbnail-button",  # Apply class to button
        )

    def create_image_modal_layout(image_filename: str, index: int, id_prefix: str):
        """Creates the modal layout content with standard styling."""
        original_filename = image_filename.replace("_optimized", "_original")
        pattern = r"(?:.*/)?(\d{8})_(\d{6})_([A-Za-z]+_[A-Za-z]+)_optimized\.jpg"
        match = re.match(pattern, image_filename)
        if match:
            date_str, time_str, class_name = match.groups()
            formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
            formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
            formatted_class = class_name.replace("_", " ")
            title_content = [
                f"{formatted_date} {formatted_time} - ",
                html.Em(formatted_class),
            ]
        else:
            title_content = image_filename

        return dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(title_content), close_button=False),
                dbc.ModalBody(
                    html.Img(
                        src=f"/images/{image_filename}",
                        className="modal-image",
                        id={"type": f"{id_prefix}-modal-image", "index": index},
                    )
                ),
                dbc.ModalFooter(
                    [
                        html.A(
                            dbc.Button("Download", color="secondary", target="_blank"),
                            href=f"/images/{original_filename}",
                            download=original_filename,
                            className="modal-download-link",
                        ),
                        dbc.Button(
                            "Close",
                            id={"type": f"{id_prefix}-close", "index": index},
                            className="ms-auto",
                            n_clicks=0,
                        ),
                    ]
                ),
            ],
            id={"type": f"{id_prefix}-modal", "index": index},
            is_open=False,
            size="lg",
        )

    def create_thumbnail_info_box(best_class, best_class_conf, top1_class, top1_conf):
        """
        Creates the standardized information box displayed below gallery thumbnails.

        Args:
            best_class (str): The scientific name from the detector.
            best_class_conf (str or float): The confidence score from the detector.
            top1_class (str): The scientific name from the classifier.
            top1_conf (str or float): The confidence score from the classifier.

        Returns:
            html.Div: The Dash HTML component for the info box.
        """
        # Basic error checking/default values for confidence if they might be missing/invalid
        try:
            detector_conf_percent = int(float(best_class_conf) * 100)
        except (ValueError, TypeError):
            detector_conf_percent = 0

        try:
            classifier_conf_percent = int(float(top1_conf) * 100)
        except (ValueError, TypeError):
            classifier_conf_percent = 0

        # Format names (handle potential None or empty strings if necessary)
        best_class_sci = best_class.replace("_", " ") if best_class else "N/A"
        top1_class_sci = (
            top1_class.replace("_", " ") if top1_class else "N/A"
        )  # We need this formatted for consistency later, even if not displayed directly in Classifier line

        common_name_display = (
            COMMON_NAMES.get(best_class, best_class_sci) if best_class else "Unknown"
        )

        return html.Div(
            [
                # Line 1: Common Name (from best_class)
                html.Span(
                    html.Strong(common_name_display), className="info-common-name"
                ),
                # Line 2: Scientific Name (from best_class)
                html.Span(
                    ["(", html.I(best_class_sci), ")"], className="info-scientific-name"
                ),
                # Line 3: Classifier Confidence (ONLY confidence percentage)
                html.Span(
                    f"Classifier: {classifier_conf_percent}%",
                    className="info-classifier-conf",
                ),
                # Line 4: Detector Confidence (WITH formatted best_class name)
                html.Span(
                    [
                        "Detector: ",
                        html.I(best_class_sci),  # Use formatted detector name
                        f" ({detector_conf_percent}%)",  # Detector confidence
                    ],
                    className="info-detector-conf",
                ),
            ],
            className="thumbnail-info",
        )

    def create_fused_agreement_info_box(final_class, combined_score, conf_d, conf_c):
        """Creates an info box for the agreement-based fusion summary."""
        final_class_sci = final_class.replace("_", " ") if final_class else "N/A"
        common_name_display = (
            COMMON_NAMES.get(final_class, final_class_sci)
            if final_class
            else "Unbekannt"
        )
        combined_score_percent = int(
            combined_score * 100
        )  # Product score needs scaling
        detector_conf_percent = int(conf_d * 100)
        classifier_conf_percent = int(conf_c * 100)

        return html.Div(
            [
                html.Span(
                    html.Strong(common_name_display), className="info-common-name"
                ),
                html.Span(
                    ["(", html.I(final_class_sci), ")"],
                    className="info-scientific-name",
                ),
                # Show the combined score prominently
                html.Span(
                    f"Product: {combined_score_percent}%",
                    className="info-combined-conf",
                ),
                # Optionally show original scores for context (smaller font?)
                html.Span(
                    f"(Det: {detector_conf_percent}%, Cls: {classifier_conf_percent}%)",
                    className="info-detector-conf",
                ),
            ],
            className="thumbnail-info",
        )

    def create_fused_weighted_info_box(
        final_class, combined_score, best_class, conf_d, conf_c
    ):
        """Creates an info box for the weighted fusion summary."""
        final_class_sci = final_class.replace("_", " ") if final_class else "N/A"
        common_name_display = (
            COMMON_NAMES.get(final_class, final_class_sci)
            if final_class
            else "Unbekannt"
        )
        combined_score_percent = int(combined_score * 100)  # Weighted score already 0-1
        detector_conf_percent = int(conf_d * 100)
        classifier_conf_percent = int(conf_c * 100)
        detector_class_sci = best_class.replace("_", " ") if best_class else "N/A"

        # Indicate if detector disagreed
        disagreement_note = ""
        if final_class != best_class and best_class:
            disagreement_note = (
                f" (Det: {detector_class_sci})"  # Show what detector thought
            )

        return html.Div(
            [
                html.Span(
                    html.Strong(common_name_display), className="info-common-name"
                ),
                html.Span(
                    ["(", html.I(final_class_sci), ")", disagreement_note],
                    className="info-scientific-name",
                ),
                # Show the combined score prominently
                html.Span(
                    f"Weighted: {combined_score_percent}%",
                    className="info-combined-conf",
                ),
                html.Span(
                    f"(Det: {detector_conf_percent}%, Cls: {classifier_conf_percent}%)",
                    className="info-regular",
                ),
            ],
            className="thumbnail-info",
        )

    def create_subgallery_modal(image_filename: str, index: int):
        return create_image_modal_layout(image_filename, index, "subgallery-modal")

    def generate_daily_fused_summary_agreement(date_str_iso, images_for_date):
        """Generates a gallery based on agreement and multiplicative score for a specific date."""
        best_results_per_species = (
            {}
        )  # key: species_name, value: (combined_score, path, best_class, conf_d, top1_class, conf_c)
        # Input images_for_date is list of: (path, best_class, best_class_conf, top1_class, top1_conf)

        for path, best_class, best_class_conf, top1_class, top1_conf in images_for_date:
            if not best_class or not top1_class:
                continue  # Need both classes

            try:
                conf_d = float(best_class_conf)
                conf_c = float(top1_conf)
            except (ValueError, TypeError):
                continue  # Skip if confidences aren't valid numbers

            is_valid = (
                conf_d >= CONFIDENCE_THRESHOLD_DETECTION
                and conf_c >= CLASSIFIER_CONFIDENCE_THRESHOLD
                and best_class == top1_class
            )

            if is_valid:
                final_class = best_class
                combined_score = conf_d * conf_c  # Multiplicative score

                # Check if this is the best score found so far for this species ON THIS DAY
                current_best_score = best_results_per_species.get(final_class, (-1.0,))[
                    0
                ]
                if combined_score > current_best_score:
                    best_results_per_species[final_class] = (
                        combined_score,
                        path,
                        best_class,
                        conf_d,
                        top1_class,
                        conf_c,
                    )

        # Convert dict to list for sorting/display
        summary_data = [
            (
                species,
                data[0],
                data[1],
                data[3],
                data[5],
            )  # (final_class, combined_score, path, conf_d, conf_c)
            for species, data in best_results_per_species.items()
        ]

        # Sort by combined score descending for daily view
        summary_data.sort(key=lambda x: x[1], reverse=True)
        summary_data = summary_data[:RECENT_IMAGES_COUNT]  # Limit count for daily view

        # --- Build Gallery ---
        gallery_items = []
        modals = []  # Keep modals separate

        if not summary_data:
            # Return placeholder component
            return html.P(
                f"No matching detections found for {date_str_iso}.",
                className="text-center text-muted small",
            )

        base_index_str = date_str_iso.replace("-", "")
        thumbnail_id_type = "daily-fused-agreement-thumbnail"
        modal_id_prefix = "daily-fused-agreement"

        for i, (final_class, combined_score, img, conf_d, conf_c) in enumerate(
            summary_data
        ):
            unique_index = f"{base_index_str}-agree-{i}"
            tile = html.Div(
                [
                    create_thumbnail_button(img, unique_index, thumbnail_id_type),
                    create_fused_agreement_info_box(
                        final_class, combined_score, conf_d, conf_c
                    ),
                ],
                className="gallery-tile",
            )
            gallery_items.append(tile)
            # Create and add modal to the separate list
            modals.append(create_image_modal_layout(img, unique_index, modal_id_prefix))

        return html.Div(gallery_items + modals, className="gallery-grid-container")

    def generate_daily_fused_summary_weighted(date_str_iso, images_for_date):
        """Generates a gallery based on weighted score for a specific date."""
        best_results_per_species = (
            {}
        )  # key: species_name (top1_class), value: (combined_score, path, best_class, conf_d, top1_class, conf_c)
        # Input images_for_date is list of: (path, best_class, best_class_conf, top1_class, top1_conf)

        for path, best_class, best_class_conf, top1_class, top1_conf in images_for_date:
            if not top1_class:
                continue  # Require a classifier result

            try:
                conf_d = float(best_class_conf) if best_class_conf else 0.0
                conf_c = float(top1_conf)
            except (ValueError, TypeError):
                continue

            is_valid = (
                conf_d >= CONFIDENCE_THRESHOLD_DETECTION
                and conf_c >= CLASSIFIER_CONFIDENCE_THRESHOLD
            )

            if is_valid:
                final_class = top1_class  # Prioritize classifier label
                combined_score = FUSION_ALPHA * conf_d + (1.0 - FUSION_ALPHA) * conf_c

                # Check if this is the best score found so far for this final_class ON THIS DAY
                current_best_score = best_results_per_species.get(final_class, (-1.0,))[
                    0
                ]
                if combined_score > current_best_score:
                    best_results_per_species[final_class] = (
                        combined_score,
                        path,
                        best_class,
                        conf_d,
                        top1_class,
                        conf_c,
                    )

        # Convert dict to list for sorting/display
        summary_data = [
            # (final_class, combined_score, path, best_class, conf_d, conf_c)
            (species, data[0], data[1], data[2], data[3], data[5])
            for species, data in best_results_per_species.items()
        ]

        # Sort by combined score descending for daily view
        summary_data.sort(key=lambda x: x[1], reverse=True)
        summary_data = summary_data[:RECENT_IMAGES_COUNT]  # Limit count for daily view

        # --- Build Gallery ---
        gallery_items = []
        modals = []  # Keep modals separate

        if not summary_data:
            # Return placeholder and empty list for modals
            return html.P(
                f"No weighted detections found for {date_str_iso}.",
                className="text-center text-muted small",
            )

        base_index_str = date_str_iso.replace("-", "")
        thumbnail_id_type = "daily-fused-weighted-thumbnail"
        modal_id_prefix = "daily-fused-weighted"

        for i, (
            final_class,
            combined_score,
            img,
            best_class_orig,
            conf_d_orig,
            conf_c_orig,
        ) in enumerate(summary_data):
            unique_index = f"{base_index_str}-weight-{i}"
            tile = html.Div(
                [
                    create_thumbnail_button(img, unique_index, thumbnail_id_type),
                    create_fused_weighted_info_box(
                        final_class,
                        combined_score,
                        best_class_orig,
                        conf_d_orig,
                        conf_c_orig,
                    ),
                ],
                className="gallery-tile",
            )
            gallery_items.append(tile)
            # Create and add modal to the separate list
            modals.append(create_image_modal_layout(img, unique_index, modal_id_prefix))

        # Return the Div containing only gallery items, and the list of modals
        return html.Div(gallery_items + modals, className="gallery-grid-container")

    def generate_all_time_detector_summary():
        """
        Generates a gallery of the best detector result for each species across all time,
        considering only detections above the confidence threshold.
        """
        now = time.time()
        cached_payload = _species_summary_cache.get("payload") or {}
        if (
            cached_payload.get("detector") is not None
            and (now - _species_summary_cache["timestamp"]) < 600
        ):
            return cached_payload["detector"]
        all_images = get_captured_images()  # Get all images (uses cache)
        best_images_all_time = {}  # Dictionary to store best image per species

        # Find the highest confidence detection for each species THAT MEETS THE THRESHOLD
        for _, path, best_class, best_class_conf, top1_class, top1_conf in all_images:
            if not best_class:
                continue  # Skip if detector class is missing
            try:
                conf_val = float(best_class_conf)
            except (ValueError, TypeError):
                continue  # Skip if confidence is not a valid number

            if conf_val < CONFIDENCE_THRESHOLD_DETECTION:
                continue  # Skip if detection confidence is below the threshold

            # Get the current best confidence stored for this species (default to -1.0 if not found)
            current_best_conf = best_images_all_time.get(
                best_class, (None, -1.0, None, None)
            )[1]

            # If the current image's confidence is higher than the stored one, update it
            if conf_val > current_best_conf:
                best_images_all_time[best_class] = (
                    path,
                    conf_val,
                    top1_class,
                    top1_conf,
                )

        # Create the list of unique detections from the dictionary
        all_time_unique_detector = [
            (
                data[0],
                species,
                data[1],
                data[2],
                data[3],
            )  # (path, best_class, conf_val, top1_class, top1_conf)
            for species, data in best_images_all_time.items()
        ]
        # Sort alphabetically by the common name of the detected species (best_class)
        all_time_unique_detector.sort(key=lambda x: COMMON_NAMES.get(x[1], x[1]))

        gallery_items, modals = [], []
        if not all_time_unique_detector:
            return html.P(
                f"No species (Detector) detected above threshold ({int(CONFIDENCE_THRESHOLD_DETECTION * 100)}%) yet.",
                className="text-center text-muted",
            )

        thumbnail_id_type = "alltime-detector-thumbnail"
        modal_id_prefix = "alltime-detector"

        # Generate gallery tiles and modals
        for i, (img, best_class, confidence, top1_class, top1_conf) in enumerate(
            all_time_unique_detector
        ):
            tile = html.Div(
                [
                    create_thumbnail_button(img, i, thumbnail_id_type),
                    create_thumbnail_info_box(
                        best_class, confidence, top1_class, top1_conf
                    ),  # Standard info box
                ],
                className="gallery-tile",
            )
            gallery_items.append(tile)
            modals.append(create_image_modal_layout(img, i, modal_id_prefix))

        payload = html.Div(gallery_items + modals, className="gallery-grid-container")
        _species_summary_cache["payload"] = _species_summary_cache["payload"] or {}
        _species_summary_cache["payload"]["detector"] = payload
        _species_summary_cache["timestamp"] = now
        return payload

    def generate_all_time_classifier_summary():
        """Generates a gallery of the best classifier result for each species across all time."""
        now = time.time()
        cached_payload = _species_summary_cache.get("payload") or {}
        if (
            cached_payload.get("classifier") is not None
            and (now - _species_summary_cache["timestamp"]) < 600
        ):
            return cached_payload["classifier"]
        all_images = get_captured_images()  # Get all images (uses cache)
        best_classifier_images_all_time = (
            {}
        )  # Dictionary to store best image per classified species

        for _, path, best_class, best_class_conf, top1_class, top1_conf in all_images:
            if not top1_class:
                continue  # Skip if classifier class is missing
            try:
                conf_val = float(top1_conf)
            except (ValueError, TypeError):
                continue

            if conf_val < CLASSIFIER_CONFIDENCE_THRESHOLD:
                continue

            current_best_conf = best_classifier_images_all_time.get(
                top1_class, (None, None, None, None, -1.0)
            )[4]
            if conf_val > current_best_conf:
                # Store original detector info too
                best_classifier_images_all_time[top1_class] = (
                    path,
                    best_class,
                    best_class_conf,
                    top1_class,
                    conf_val,
                )

        all_time_unique_classifier = [
            (data[0], data[1], data[2], species, data[4])
            for species, data in best_classifier_images_all_time.items()
        ]
        # Sort by common name of the CLASSIFIED species (top1_class)
        all_time_unique_classifier.sort(key=lambda x: COMMON_NAMES.get(x[3], x[3]))

        gallery_items, modals = [], []
        if not all_time_unique_classifier:
            return html.P(
                f"No species (Classifier) detected yet (Threshold: {int(CLASSIFIER_CONFIDENCE_THRESHOLD * 100)}%).",
                className="text-center text-muted",
            )

        thumbnail_id_type = "alltime-classifier-thumbnail"
        modal_id_prefix = "alltime-classifier"

        for i, (img, best_class, best_class_conf, top1_class, top1_conf) in enumerate(
            all_time_unique_classifier
        ):
            tile = html.Div(
                [
                    create_thumbnail_button(img, i, thumbnail_id_type),
                    create_thumbnail_info_box(
                        best_class, best_class_conf, top1_class, top1_conf
                    ),  # Standard info box
                ],
                className="gallery-tile",
            )
            gallery_items.append(tile)
            modals.append(create_image_modal_layout(img, i, modal_id_prefix))

        payload = html.Div(gallery_items + modals, className="gallery-grid-container")
        _species_summary_cache["payload"] = _species_summary_cache["payload"] or {}
        _species_summary_cache["payload"]["classifier"] = payload
        _species_summary_cache["timestamp"] = now
        return payload

    def generate_all_time_fused_summary_agreement():
        """Generates a gallery based on agreement and multiplicative score."""
        now = time.time()
        cached_payload = _species_summary_cache.get("payload") or {}
        if (
            cached_payload.get("agreement") is not None
            and (now - _species_summary_cache["timestamp"]) < 600
        ):
            return cached_payload["agreement"]
        all_images = get_captured_images()
        best_results_per_species = (
            {}
        )  # key: species_name, value: (combined_score, path, best_class, conf_d, top1_class, conf_c)

        for _, path, best_class, best_class_conf, top1_class, top1_conf in all_images:
            if not best_class or not top1_class:
                continue  # Need both classes

            try:
                conf_d = float(best_class_conf)
                conf_c = float(top1_conf)
            except (ValueError, TypeError):
                continue  # Skip if confidences aren't valid numbers

            is_valid = (
                conf_d >= CONFIDENCE_THRESHOLD_DETECTION
                and conf_c >= CLASSIFIER_CONFIDENCE_THRESHOLD
                and best_class == top1_class
            )

            if is_valid:
                final_class = best_class
                combined_score = conf_d * conf_c  # Multiplicative score

                # Check if this is the best score found so far for this species
                current_best_score = best_results_per_species.get(final_class, (-1.0,))[
                    0
                ]
                if combined_score > current_best_score:
                    best_results_per_species[final_class] = (
                        combined_score,
                        path,
                        best_class,
                        conf_d,
                        top1_class,
                        conf_c,
                    )

        # Convert dict to list for sorting/display
        # List items: (final_class, combined_score, path, conf_d, conf_c) <= simplified for info box
        summary_data = [
            (species, data[0], data[1], data[3], data[5])
            for species, data in best_results_per_species.items()
        ]

        # Sort alphabetically by common name of the agreed class
        summary_data.sort(key=lambda x: COMMON_NAMES.get(x[0], x[0]))

        # --- Build Gallery ---
        gallery_items = []
        modals = []
        if not summary_data:
            return html.P(
                "No matching detections found above thresholds.",
                className="text-center text-muted",
            )

        thumbnail_id_type = "alltime-fused-agreement-thumbnail"
        modal_id_prefix = "alltime-fused-agreement"

        for i, (final_class, combined_score, img, conf_d, conf_c) in enumerate(
            summary_data
        ):
            tile = html.Div(
                [
                    create_thumbnail_button(img, i, thumbnail_id_type),
                    create_fused_agreement_info_box(
                        final_class, combined_score, conf_d, conf_c
                    ),  # Use specific info box
                ],
                className="gallery-tile",
            )
            gallery_items.append(tile)
            modals.append(
                create_image_modal_layout(img, i, modal_id_prefix)
            )  # Standard modal layout

        payload = html.Div(gallery_items + modals, className="gallery-grid-container")
        _species_summary_cache["payload"] = _species_summary_cache["payload"] or {}
        _species_summary_cache["payload"]["agreement"] = payload
        _species_summary_cache["timestamp"] = now
        return payload

    def generate_all_time_fused_summary_weighted():
        """Generates a gallery based on weighted score, prioritizing classifier label."""
        now = time.time()
        cached_payload = _species_summary_cache.get("payload") or {}
        if (
            cached_payload.get("weighted") is not None
            and (now - _species_summary_cache["timestamp"]) < 600
        ):
            return cached_payload["weighted"]
        all_images = get_captured_images()
        best_results_per_species = (
            {}
        )  # key: species_name (top1_class), value: (combined_score, path, best_class, conf_d, top1_class, conf_c)

        for _, path, best_class, best_class_conf, top1_class, top1_conf in all_images:
            # Require a classifier result for this method
            if not top1_class:
                continue

            try:
                conf_d = (
                    float(best_class_conf) if best_class_conf else 0.0
                )  # Default detector conf to 0 if missing
                conf_c = float(top1_conf)
            except (ValueError, TypeError):
                continue  # Skip if classifier conf is invalid

            is_valid = (
                conf_d >= CONFIDENCE_THRESHOLD_DETECTION
                and conf_c >= CLASSIFIER_CONFIDENCE_THRESHOLD
            )

            if is_valid:
                final_class = top1_class  # Prioritize classifier label
                combined_score = (
                    FUSION_ALPHA * conf_d + (1.0 - FUSION_ALPHA) * conf_c
                )  # Weighted score

                # Check if this is the best score found so far for this final_class (top1_class)
                current_best_score = best_results_per_species.get(final_class, (-1.0,))[
                    0
                ]
                if combined_score > current_best_score:
                    # Store original detector info as well
                    best_results_per_species[final_class] = (
                        combined_score,
                        path,
                        best_class,
                        conf_d,
                        top1_class,
                        conf_c,
                    )

        # Convert dict to list for sorting/display
        # List items: (final_class, combined_score, path, best_class, conf_d, conf_c) <= Need original detector class for info box
        summary_data = [
            (species, data[0], data[1], data[2], data[3], data[5])
            for species, data in best_results_per_species.items()
        ]

        # Sort alphabetically by common name of the FINAL class (top1_class)
        summary_data.sort(key=lambda x: COMMON_NAMES.get(x[0], x[0]))

        # --- Build Gallery ---
        gallery_items = []
        modals = []
        if not summary_data:
            return html.P(
                "No detections found above thresholds for weighted evaluation.",
                className="text-center text-muted",
            )

        thumbnail_id_type = "alltime-fused-weighted-thumbnail"
        modal_id_prefix = "alltime-fused-weighted"

        for i, (
            final_class,
            combined_score,
            img,
            best_class,
            conf_d,
            conf_c,
        ) in enumerate(summary_data):
            tile = html.Div(
                [
                    create_thumbnail_button(img, i, thumbnail_id_type),
                    create_fused_weighted_info_box(
                        final_class, combined_score, best_class, conf_d, conf_c
                    ),  # Use specific info box
                ],
                className="gallery-tile",
            )
            gallery_items.append(tile)
            modals.append(
                create_image_modal_layout(img, i, modal_id_prefix)
            )  # Standard modal layout

        payload = html.Div(gallery_items + modals, className="gallery-grid-container")
        _species_summary_cache["payload"] = _species_summary_cache["payload"] or {}
        _species_summary_cache["payload"]["weighted"] = payload
        _species_summary_cache["timestamp"] = now
        return payload

    def species_summary_layout():
        """Layout for the all-time species summary page."""
        detector_summary = dcc.Loading(
            type="circle", children=generate_all_time_detector_summary()
        )
        classifier_summary = dcc.Loading(
            type="circle", children=generate_all_time_classifier_summary()
        )
        fused_agreement_summary = dcc.Loading(
            type="circle", children=generate_all_time_fused_summary_agreement()
        )
        fused_weighted_summary = dcc.Loading(
            type="circle", children=generate_all_time_fused_summary_weighted()
        )

        return dbc.Container(
            [
                generate_navbar(),  # Include navbar
                html.H1("Species Summary (All Days)", className="text-center my-3"),
                # --- Section 1: Best Detector ---
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3(
                                    "Best Detection per Species",
                                    className="text-center mt-4 mb-3",
                                ),
                                html.P(
                                    f"Shows the image with the highest detector confidence (>= {int(CONFIDENCE_THRESHOLD_DETECTION*100)}%) for each classified species.",
                                    className="text-center text-muted small mb-3",
                                ),
                                detector_summary,
                            ],
                            width=12,
                        ),
                    ]
                ),
                html.Hr(className="my-4"),
                # --- Section 2: Best Classifier ---
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3(
                                    "Best Classification per Species",
                                    className="text-center mt-4 mb-3",
                                ),
                                html.P(
                                    f"Shows the image with the highest classifier confidence (>= {int(CLASSIFIER_CONFIDENCE_THRESHOLD*100)}%) for each classified species.",
                                    className="text-center text-muted small mb-3",
                                ),
                                classifier_summary,
                            ],
                            width=12,
                        ),
                    ]
                ),
                html.Hr(className="my-4"),
                # --- Section 3: Fused - Agreement & Multiplicative Score ---
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3(
                                    "Agreement & Product Score",
                                    className="text-center mt-4 mb-3",
                                ),
                                html.P(
                                    f"Shows the image with the highest product score (Detector x Classifier), only if both models agree and are above their thresholds (Det >= {int(CONFIDENCE_THRESHOLD_DETECTION*100)}%, Cls >= {int(CLASSIFIER_CONFIDENCE_THRESHOLD*100)}%).",
                                    className="text-center text-muted small mb-3",
                                ),
                                fused_agreement_summary,
                            ],
                            width=12,
                        ),
                    ]
                ),
                html.Hr(className="my-4"),
                # --- Section 4: Fused - Weighted Score ---
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3(
                                    f"Weighted Score, α={FUSION_ALPHA}",
                                    className="text-center mt-4 mb-3",
                                ),
                                html.P(
                                    f"Shows the image with the highest weighted score ({FUSION_ALPHA*100:.0f}% Detector + { (1-FUSION_ALPHA)*100:.0f}% Classifier), if both models are above their thresholds. In case of disagreement, the classifier's class is used.",
                                    className="text-center text-muted small mb-3",
                                ),
                                fused_weighted_summary,
                            ],
                            width=12,
                        ),
                    ]
                ),
            ],
            fluid=True,
        )

    # -------------------------------------
    # Edit Page Layout Generation
    # -------------------------------------
    def generate_edit_page(date_str_iso):
        """Generates the layout for the image editing page (simplified)."""
        df = read_csv_for_date(date_str_iso)
        if df.empty:
            return dbc.Container(
                [
                    generate_navbar(),
                    html.H2(
                        f"Edit Images from {date_str_iso}", className="text-center my-3"
                    ),
                    dbc.Alert(
                        f"No images found or error reading data for {date_str_iso}.",
                        color="warning",
                    ),
                    dbc.Button(
                        "Back to Subgallery",
                        href=f"/gallery/{date_str_iso}",
                        color="secondary",
                        className="me-2",
                    ),
                    dbc.Button(
                        "Back to Main Gallery", href="/gallery", color="secondary"
                    ),
                ],
                fluid=True,
            )

        image_tiles = build_edit_tiles(date_str_iso, df)

        return dbc.Container(
            [
                generate_navbar(),
                html.H2(
                    f"Edit Images for {date_str_iso}", className="text-center my-3"
                ),
                # Navigation Buttons
                html.Div(
                    [
                        dbc.Button(
                            "Back to Subgallery",
                            href=f"/gallery/{date_str_iso}",
                            color="secondary",
                            outline=True,
                            className="me-2",
                        ),
                        dbc.Button(
                            "Back to Main Gallery",
                            href="/gallery",
                            color="secondary",
                            outline=True,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Select all on this page",
                                id="select-all-button",
                                color="primary",
                                outline=True,
                                className="me-2",
                                n_clicks=0,
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Clear selection",
                                id="clear-selection-button",
                                color="secondary",
                                outline=True,
                                className="me-2",
                                n_clicks=0,
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Div("Selected: 0", id="selected-count"),
                            width="auto",
                            className="align-self-center",
                        ),
                    ],
                    className="mb-3 align-items-center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Filter by download status", className="fw-semibold"),
                                dcc.Dropdown(
                                    id="edit-filter-status",
                                    options=[
                                        {"label": "All", "value": "all"},
                                        {"label": "Downloaded only", "value": "downloaded"},
                                        {"label": "Not downloaded", "value": "not_downloaded"},
                                    ],
                                    value="all",
                                    clearable=False,
                                    style={"minWidth": "220px"},
                                ),
                            ],
                            width="auto",
                            className="me-3",
                        ),
                        dbc.Col(
                            [
                                html.Label("Sort", className="fw-semibold"),
                                dcc.Dropdown(
                                    id="edit-sort",
                                    options=[
                                        {"label": "Newest first", "value": "time_desc"},
                                        {"label": "Oldest first", "value": "time_asc"},
                                        {"label": "Species (A→Z)", "value": "species"},
                                    ],
                                    value="time_desc",
                                    clearable=False,
                                    style={"minWidth": "180px"},
                                ),
                            ],
                            width="auto",
                            className="me-3",
                        ),
                        dbc.Col(
                            [
                                html.Label("Detector min conf", className="fw-semibold"),
                                dbc.Input(
                                    id="edit-filter-detector-min",
                                    type="number",
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    value=0,
                                    style={"maxWidth": "120px"},
                                ),
                            ],
                            width="auto",
                            className="me-3",
                        ),
                        dbc.Col(
                            [
                                html.Label("Classifier max conf", className="fw-semibold"),
                                dbc.Input(
                                    id="edit-filter-classifier-min",
                                    type="number",
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    value=0,
                                    style={"maxWidth": "140px"},
                                ),
                            ],
                            width="auto",
                            className="me-3",
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Apply filter",
                                id="edit-filter-apply",
                                color="secondary",
                                outline=True,
                                className="mt-4",
                                n_clicks=0,
                            ),
                            width="auto",
                            className="align-self-end",
                        ),
                    ],
                    className="mb-3 align-items-center",
                ),
                # Action Buttons, Confirmation, Store, Download
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H5("Delete"),
                                html.P(
                                    "Removes selected images and their DB rows. This cannot be undone.",
                                    className="text-muted small",
                                ),
                                dbc.Button(
                                    "Delete Selected Images",
                                    id="delete-button",
                                    color="danger",
                                    className="me-2",
                                ),
                            ],
                            width="auto",
                        ),
                        dbc.Col(
                            [
                                html.H5("Download"),
                                html.P(
                                    "Download originals for selected images.",
                                    className="text-muted small",
                                ),
                                dbc.Button(
                                    "Download Selected Images",
                                    id="download-button",
                                    color="success",
                                ),
                            ],
                            width="auto",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Span("Selected: ", className="fw-semibold me-1"),
                                        html.Span("0", id="selected-summary-count"),
                                        html.Span(" | ", className="mx-1"),
                                        html.Span("None", id="selected-summary-list"),
                                    ],
                                    className="text-muted small",
                                )
                            ],
                            width="auto",
                        ),
                    ],
                    justify="start",
                    className="mb-3",
                ),
                html.Div(
                    f"Page 1 (static edit view for {date_str_iso})",
                    className="text-muted small mb-2",
                ),
                dcc.ConfirmDialog(
                    id="confirm-delete",
                    message="Are you sure you want to permanently delete the selected images? This cannot be undone.",
                ),
                dcc.Store(
                    id="selected-images-store", data=[]
                ),  # Store is now updated by Python callback
                dcc.Download(id="download-zip"),
                html.Div(id="edit-status-message"),
                # The Grid containing the tiles
                html.Div(
                    image_tiles,
                    className="gallery-grid-container",
                    id="edit-gallery-grid",
                ),  # Keep ID for JS
                # Bottom Action Buttons
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Delete Selected Images",
                                id="delete-button-bottom",
                                color="danger",
                                className="me-2",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Download Selected Images",
                                    id="download-button-bottom",
                                    color="success",
                                ),
                            ],
                            width="auto",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Span("Selected: ", className="fw-semibold me-1"),
                                        html.Span("0", id="selected-summary-count-bottom"),
                                        html.Span(" | ", className="mx-1"),
                                        html.Span("None", id="selected-summary-list-bottom"),
                                    ],
                                    className="text-muted small",
                                )
                            ],
                            width="auto",
                        ),
                    ],
                    justify="start",
                    className="mt-3",
                ),
            ],
            fluid=True,
            id="edit-page-container",
        )

    def generate_navbar():
        """Creates the navbar with the logo for gallery pages."""
        today_date_iso = datetime.now().strftime("%Y-%m-%d")
        return dbc.NavbarSimple(
            brand=html.Img(
                src="/assets/WatchMyBirds.png",
                className="img-fluid round-logo",
            ),
            brand_href="/",
            children=[
                dbc.NavItem(dbc.NavLink("Live Stream", href="/", className="mx-auto")),
                dbc.NavItem(
                    dbc.NavLink("Gallery", href="/gallery", className="mx-auto")
                ),
                dbc.NavItem(
                    dbc.NavLink("Today", href=f"/gallery/{today_date_iso}", className="mx-auto")
                ),
                dbc.NavItem(
                    dbc.NavLink("Species Summary", href="/species", className="mx-auto")
                ),
                dbc.NavItem(
                    dbc.NavLink("Settings", href="/settings", className="mx-auto")
                ),
            ],
            color="primary",
            dark=True,
            fluid=True,
            className="justify-content-center custom-navbar",
        )

    def generate_gallery():
        """Generates the main gallery page with daily subgallery links, including the logo."""
        daily_covers = get_daily_covers()
        sorted_dates = sorted(daily_covers.keys(), reverse=True)

        grid_items = []
        if not sorted_dates:
            grid_items.append(
                html.P(
                    "No images in the gallery yet.",
                    className="text-center text-muted mt-5",
                )
            )
        else:
            for date in sorted_dates:
                thumbnail_rel_path = daily_covers.get(date)
                if not thumbnail_rel_path:
                    continue
                # Derive the zoomed filename for the thumbnail link display
                thumbnail_display_path = derive_zoomed_filename(thumbnail_rel_path)

                grid_items.append(
                    html.Div(
                        [
                            html.A(
                                html.Img(
                                    src=f"/images/{thumbnail_display_path}",
                                    className="main-gallery-image",
                                    style={"width": f"{IMAGE_WIDTH}px"},
                                ),
                                href=f"/gallery/{date}",  # Link to subgallery
                            ),
                            html.P(date, className="main-gallery-date"),
                        ],
                        className="main-gallery-item",
                    )
                )

        content = html.Div(grid_items, className="gallery-grid-container")
        return dbc.Container(
            [
                generate_navbar(),
                html.H1("Gallery", className="text-center my-3"),
                content,  # Add the container with grid items
            ],
            fluid=True,
        )

    def generate_subgallery(date, page=1, include_header=True):
        """
        Generates the content for a specific date's subgallery page,
        including daily species summaries, pagination, loading indicator,
        and empty state handling.
        """
        # Get today's date for comparison >>> ---
        today_date_iso = datetime.now().strftime("%Y-%m-%d")
        rows = fetch_day_images(db_conn, date)
        df = _rows_to_df(rows)
        images_for_summary = []
        for row in rows:
            optimized_name = row["optimized_name"]
            if not optimized_name:
                continue
            rel_path = os.path.join(date.replace("-", ""), optimized_name)
            item = (
                rel_path,
                row["best_class"],
                row["best_class_conf"],
                row["top1_class_name"],
                row["top1_confidence"],
            )
            images_for_summary.append(item)

        if (
            df.empty and not images_for_summary
        ):  # Check both in case one fails but not the other
            # Handle case where no data exists for the date
            return dbc.Container(
                [
                    generate_navbar(),
                    html.H2(f"Images from {date}", className="text-center"),
                    dbc.Alert(f"No images found for {date}.", color="info"),
                    dbc.Button(
                        "Back to Gallery Overview",
                        href="/gallery",
                        color="secondary",
                        outline=True,
                    ),
                ],
                fluid=True,
            )

        images_for_this_date = images_for_summary
        total_images = fetch_day_count(db_conn, date)
        total_pages = math.ceil(total_images / PAGE_SIZE) or 1
        page = max(1, min(page, total_pages))
        start_index = (page - 1) * PAGE_SIZE
        end_index = page * PAGE_SIZE
        # Slice the DataFrame for pagination
        page_df = df.iloc[start_index:end_index]
        page_images = images_for_this_date[start_index:end_index]

        # --- Pagination Controls ---
        page_links = []
        for p in range(1, total_pages + 1):
            if p == page:
                link = dbc.Button(
                    str(p), color="primary", disabled=True, className="mx-1", size="sm"
                )
            else:
                link = dbc.Button(
                    str(p),
                    color="secondary",
                    href=f"/gallery/{date}?page={p}",
                    className="mx-1",
                    size="sm",
                )
            page_links.append(link)

        # --- Define Top Controls ---
        pagination_controls_top = html.Div(
            page_links,  # Use the list of links
            className="pagination-controls pagination-top",
            id="pagination-top",  # Keep this ID for the scroll target
        )

        # --- Define Bottom Controls ---
        # Create a new Div, potentially cloning the links or just reusing the list
        pagination_controls_bottom = html.Div(
            page_links,  # Reuse the list of links
            className="pagination-controls pagination-bottom",
            # id="pagination-bottom" # Optional: Add a UNIQUE ID if needed, otherwise omit
        )

        # --- Main Paginated Gallery Items and Modals ---
        gallery_items = []
        subgallery_modals = []

        # Define ID types/prefixes needed within the loop
        button_id_type = "subgallery-thumbnail"
        modal_id_prefix = "subgallery-modal"

        # Iterate over the DataFrame slice >>> ---
        if page_df.empty:
            gallery_items.append(
                html.P(
                    f"No images found for {date} on this page.",
                    className="text-center text-muted mt-4 mb-4",
                )
            )
        else:
            for i, row in page_df.iterrows():
                # Construct the relative path from date and optimized_name
                relative_path = os.path.join(
                    date.replace("-", ""), row["optimized_name"]
                )
                # Get other data from the row
                best_class = row.get("best_class", "")
                best_class_conf = row.get("best_class_conf", "")
                top1_class = row.get("top1_class_name", "")
                top1_conf = row.get("top1_confidence", "")
                downloaded_ts = row.get(
                    "downloaded_timestamp", ""
                )  # Get download status

                unique_subgallery_index = f"{date.replace('-', '')}-sub-{start_index + i}"  # Adjust index calculation slightly if needed based on iloc behavior vs enumerate

                # Add class if downloaded
                tile_classname = "gallery-tile"
                if (
                    downloaded_ts and str(downloaded_ts).strip()
                ):  # Check if timestamp exists and is not empty/whitespace
                    tile_classname += " downloaded-image"

                info_box = create_thumbnail_info_box(
                    best_class, best_class_conf, top1_class, top1_conf
                )
                # Add downloaded text to info box
                if downloaded_ts and str(downloaded_ts).strip():
                    info_box.children.append(
                        html.Span(
                            f"Downloaded",
                            className="info-download-status text-success small",
                        )
                    )

                tile = html.Div(
                    [
                        create_thumbnail_button(
                            relative_path, unique_subgallery_index, button_id_type
                        ),
                        info_box,  # Use the potentially modified info_box
                    ],
                    className=tile_classname,
                )  # Use the potentially modified classname

                gallery_items.append(tile)
                # Use relative_path for creating modals as well
                subgallery_modals.append(
                    create_subgallery_modal(relative_path, unique_subgallery_index)
                )

        # --- Main Paginated Content Area with Loading ---
        gallery_grid = html.Div(gallery_items, className="gallery-grid-container")
        loading_wrapper = dcc.Loading(
            id=f"loading-subgallery-{date}-{page}", type="circle", children=gallery_grid
        )

        # --- Final Page Structure ---
        container_elements = []
        if include_header:
            container_elements.append(generate_navbar())
            # Group Back and Edit buttons
            header_buttons = [
                dbc.Button(
                    "Zurück zur Galerieübersicht",
                    href="/gallery",
                    color="secondary",
                    className="mb-3 mt-3 me-2",
                    outline=True,
                )
            ]
            # Conditionally add Edit Button
            if date != today_date_iso:
                # Button triggers modal, ID includes the date
                header_buttons.append(
                    dbc.Button(
                        "Edit This Day",
                        id={
                            "type": "open-edit-modal-button",
                            "date": date,
                        },  # New ID pattern
                        color="warning",
                        size="sm",
                        className="mb-3 mt-3",
                        n_clicks=0,
                    )
                )
            else:
                header_buttons.append(
                    dbc.Button(
                        "Edit (Past Days Only)",
                        color="warning",
                        size="sm",
                        className="mb-3 mt-3",
                        disabled=True,
                    )
                )
            container_elements.append(html.Div(header_buttons))

            container_elements.append(
                html.H2(f"Images from {date}", className="text-center")
            )
            container_elements.append(
                html.P(
                    f"Page {page} of {total_pages} ({total_images} images total)",
                    className="text-center text-muted small",
                )
            )

        # --- Add Daily Summary Content ONLY if page is 1 ---
        if page == 1 and include_header:
            agreement_summary_content = generate_daily_fused_summary_agreement(
                date, images_for_this_date
            )
            weighted_summary_content = generate_daily_fused_summary_weighted(
                date, images_for_this_date
            )
            container_elements.extend(
                [
                    html.Hr(),
                    html.H4(
                        "Daily Summary: Agreement & Product Score",
                        className="text-center mt-4",
                    ),
                    agreement_summary_content,
                    html.H4(
                        f"Daily Summary: Weighted Score, α={FUSION_ALPHA}",
                        className="text-center mt-4",
                    ),
                    weighted_summary_content,
                    html.Hr(),
                ]
            )

        # Add Header for the main paginated gallery section
        # container_elements.append(html.H4("Alle Bilder", className="text-center mt-4"))

        # Add pagination and main loading wrapper
        container_elements.extend(
            [pagination_controls_top, loading_wrapper, pagination_controls_bottom]
        )

        # Collect ALL Modals
        all_modals = subgallery_modals
        container_elements.extend(all_modals)

        return dbc.Container(container_elements, fluid=True)

    def generate_video_feed():
        # Load placeholder once
        static_placeholder_path = "assets/static_placeholder.jpg"
        if os.path.exists(static_placeholder_path):
            static_placeholder = cv2.imread(static_placeholder_path)
            if static_placeholder is not None:
                original_h, original_w = static_placeholder.shape[:2]
                ratio = original_h / float(original_w)
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * ratio)
                static_placeholder = cv2.resize(
                    static_placeholder, (placeholder_w, placeholder_h)
                )
            else:
                placeholder_w = output_resize_width
                placeholder_h = int(placeholder_w * 9 / 16)
                static_placeholder = np.zeros(
                    (placeholder_h, placeholder_w, 3), dtype=np.uint8
                )
        else:
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * 9 / 16)
            static_placeholder = np.zeros(
                (placeholder_h, placeholder_w, 3), dtype=np.uint8
            )

        # Parameters for text overlay
        padding_x_percent = 0.005
        padding_y_percent = 0.04
        min_font_size = 12
        min_font_size_percent = 0.05

        while True:
            start_time = time.time()
            # Retrieve the most recent display frame (raw or optimized)
            frame = detection_manager.get_display_frame()
            if frame is not None:
                # Derive size from the current frame to avoid blocking on VideoCapture availability.
                h, w = frame.shape[:2]
                output_resize_height = int(h * output_resize_width / w) if w else h
                resized_frame = cv2.resize(
                    frame, (output_resize_width, output_resize_height)
                )
                # Overlay current timestamp
                pil_image = Image.fromarray(
                    cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                )
            else:
                # If no frame, use the placeholder.
                pil_image = Image.fromarray(
                    cv2.cvtColor(static_placeholder, cv2.COLOR_BGR2RGB)
                )

            draw = ImageDraw.Draw(pil_image)
            img_width, img_height = pil_image.size
            padding_x = int(img_width * padding_x_percent)
            padding_y = int(img_height * padding_y_percent)
            custom_font = ImageFont.load_default()
            timestamp_text = time.strftime("%Y-%m-%d %H:%M:%S")
            bbox = draw.textbbox((0, 0), timestamp_text, font=custom_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = img_width - text_width - padding_x
            text_y = img_height - text_height - padding_y
            draw.text((text_x, text_y), timestamp_text, font=custom_font, fill="white")
            # Convert back to OpenCV BGR format
            frame_with_timestamp = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode(
                ".jpg", frame_with_timestamp, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            elapsed = time.time() - start_time
            stream_fps = config.get("STREAM_FPS", 0)
            if stream_fps and stream_fps > 0:
                desired_frame_time = 1.0 / stream_fps
                if elapsed < desired_frame_time:
                    time.sleep(desired_frame_time - elapsed)

    def generate_hourly_detection_plot():
        """
        Generates a Plotly bar plot showing number of detections per hour for today.
        Handles empty data gracefully and is compatible with Plotly.
        """
        # Get all captured images
        today_str = datetime.now().strftime("%Y-%m-%d")

        # Initialize count for all 24 hours
        counts = {f"{hour:02d}": 0 for hour in range(24)}

        # Fetch aggregated counts from DB
        rows = fetch_hourly_counts(db_conn, today_str)
        for row in rows:
            hour = row["hour"]
            if hour in counts:
                counts[hour] = row["count"]

        # Create lists for plotting
        hours = list(counts.keys())
        values = [counts[h] for h in hours]

        # Create a Plotly bar plot
        fig = px.bar(
            x=hours,
            y=values,
            labels={"x": "Hour of the Day", "y": "Number of Observations"},
            color_discrete_sequence=["#B5EAD7"],
        )

        fig.update_layout(
            title={
                "text": "Today's Observations per Hour",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(family="Arial, sans-serif", size=12, color="#333333"),
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=50, b=40),
            hoverlabel=dict(
                bgcolor="white", font_size=12, font_family="Arial, sans-serif"
            ),
        )

        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="#cccccc",
            tickfont=dict(color="#555555", size=11),
            gridcolor="#eeeeee",
            showgrid=True,
        )
        fig.update_yaxes(
            showline=True,
            linewidth=1,
            linecolor="#cccccc",
            tickfont=dict(color="#555555", size=11),
            gridcolor="#eeeeee",
            showgrid=False,
        )

        # Customize hover text
        fig.update_traces(
            hovertemplate="<b>Hour %{x}</b><br>Observations: %{y}<extra></extra>"
        )

        # Add a simple check for no data
        if not any(values):
            fig.update_layout(
                xaxis_showticklabels=False,
                yaxis_showticklabels=False,
                annotations=[
                    dict(
                        text="No observations today yet",
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16, color="#888888"),
                        x=0.5,
                        y=0.5,
                    )
                ],
            )

        return dcc.Loading(
            type="circle",
            children=dcc.Graph(figure=fig, config={"displayModeBar": False}),
        )

    # -----------------------------
    # Flask Server and Routes
    # -----------------------------
    # Create Flask server without overriding the static asset defaults.
    server = Flask(__name__)

    def setup_web_routes(app_server):
        # Route to serve images from the output directory.
        def serve_image(filename):
            image_path = os.path.join(output_dir, filename)
            if not os.path.exists(image_path):
                return "Image not found", 404
            return send_from_directory(output_dir, filename)

        def daily_species_summary_route():
            date_iso = request.args.get("date")
            if not date_iso:
                date_iso = datetime.now().strftime("%Y-%m-%d")
            try:
                datetime.strptime(date_iso, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid date format, expected YYYY-MM-DD"}), 400
            summary = get_daily_species_summary(date_iso)
            return jsonify({"date": date_iso, "summary": summary})

        app_server.route("/images/<path:filename>")(serve_image)
        def video_feed_route():
            return Response(
                generate_video_feed(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        def stream_status_route():
            return (
                json.dumps(
                    {
                        "video_capture_initialized": detection_manager.video_capture
                        is not None,
                        "last_frame_timestamp": detection_manager.latest_raw_timestamp
                        if hasattr(detection_manager, "latest_raw_timestamp")
                        else None,
                    }
                ),
                200,
                {"Content-Type": "application/json"},
            )

        def settings_get_route():
            return jsonify(get_settings_payload())

        def settings_post_route():
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return jsonify({"error": "Invalid payload"}), 400
            valid, errors = validate_runtime_updates(payload)
            if errors:
                return jsonify({"errors": errors}), 400
            update_runtime_settings(valid)
            return jsonify(get_settings_payload())

        app_server.add_url_rule(
            "/video_feed", endpoint="video_feed", view_func=video_feed_route
        )
        app_server.add_url_rule(
            "/stream_status", endpoint="stream_status", view_func=stream_status_route
        )
        app_server.add_url_rule(
            "/api/daily_species_summary",
            endpoint="daily_species_summary",
            view_func=daily_species_summary_route,
            methods=["GET"],
        )
        app_server.add_url_rule(
            "/api/settings", endpoint="settings_get", view_func=settings_get_route, methods=["GET"]
        )
        app_server.add_url_rule(
            "/api/settings", endpoint="settings_post", view_func=settings_post_route, methods=["POST"]
        )

    setup_web_routes(server)

    # -------------------------------------
    # Dash App Setup and Google Analytics Integration
    # -------------------------------------
    external_stylesheets = [dbc.themes.BOOTSTRAP]

    # --- Define the custom HTML structure for Dash ---
    custom_index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title> 
            {{%favicon%}}
            {{%css%}}
            </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
            </footer>
        </body>
    </html>
    """

    # --- Initialize Dash App with the custom index_string ---
    app = Dash(
        __name__,
        server=server,
        external_stylesheets=external_stylesheets,
        assets_folder=os.path.join(os.getcwd(), "assets"),
        assets_url_path="/assets",
        index_string=custom_index_string,
    )
    app.config.suppress_callback_exceptions = True

    def stream_layout():
        """Layout for the live stream page using CSS classes."""
        # --- Optional lightweight stats for today ---
        today_date_iso = datetime.now().strftime("%Y-%m-%d")
        today_count = fetch_day_count(db_conn, today_date_iso)
        hourly_chart = generate_hourly_detection_plot()
        latest_strip = []
        try:
            rows = fetch_day_images(db_conn, today_date_iso)
            for row in rows:
                optimized_name = row["optimized_name"]
                if not optimized_name:
                    continue
                rel_path = os.path.join(today_date_iso.replace("-", ""), optimized_name)
                latest_strip.append(
                    (
                        rel_path,
                        row["best_class"],
                        row["best_class_conf"],
                        row["top1_class_name"],
                        row["top1_confidence"],
                    )
                )
                if len(latest_strip) >= 5:
                    break
        except Exception as e:
            logger.error(f"Error building latest strip for today: {e}")
        latest_strip_modals = [
            create_image_modal_layout(rel_path, i, "latest-strip")
            for i, (
                rel_path,
                best_class,
                best_class_conf,
                top1_class,
                top1_conf,
            ) in enumerate(latest_strip)
        ]

        # --- Build the Layout List ---
        layout_children = [
            generate_navbar(),
            dbc.Row(
                dbc.Col(
                    html.H1(
                        f"Live Stream (Today's Detections: {today_count})",
                        className="text-center",
                    ),
                    width=12,
                    className="my-3",
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Loading(
                        id="loading-video",
                        type="default",
                        children=html.Img(
                            id="video-feed",
                            src="/video_feed",
                            className="video-feed-image",
                        ),
                    ),
                    width=12,
                ),
                className="my-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4(
                                                "Latest Detections (Top 5)",
                                                className="text-center mt-4",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                            [
                                                create_thumbnail_button(
                                                    rel_path, i, "latest-strip-thumbnail"
                                                ),
                                                create_thumbnail_info_box(
                                                    best_class,
                                                    best_class_conf,
                                                    top1_class,
                                                    top1_conf,
                                                ),
                                            ],
                                            className="gallery-tile",
                                        )
                                        for i, (
                                            rel_path,
                                            best_class,
                                            best_class_conf,
                                            top1_class,
                                            top1_conf,
                                        ) in enumerate(latest_strip)
                                    ],
                                    className="gallery-grid-container",
                                )
                                if latest_strip
                                else html.P(
                                    "No detections yet today.",
                                    className="text-center text-muted",
                                ),
                            ]
                        ),
                        width=12,
                    )
                ],
                className="my-3",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        html.H4(
                            "Today's Daily Summary (Gallery style)",
                            className="text-center mt-4",
                        ),
                        dcc.Loading(
                            type="circle",
                            children=html.Div(
                                html.P(
                                    "Loading daily summary...",
                                    className="text-center text-muted",
                                ),
                                id="today-daily-summary",
                            ),
                        ),
                        dcc.Interval(
                            id="today-daily-summary-trigger",
                            interval=700,
                            n_intervals=0,
                            max_intervals=1,
                        ),
                    ],
                    width=12,
                ),
                className="my-3",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        html.H4(
                            "Today's Species Summary",
                            className="text-center mt-4",
                        ),
                        dcc.Loading(
                            type="circle",
                            children=html.Div(
                                html.P(
                                    "Loading species summary...",
                                    className="text-center text-muted",
                                ),
                                id="today-species-summary",
                            ),
                        ),
                        dcc.Interval(
                            id="today-species-summary-trigger",
                            interval=500,
                            n_intervals=0,
                            max_intervals=1,
                        ),
                    ],
                    width=12,
                ),
                className="my-3",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        html.H4("Today - Hourly Detections", className="text-center"),
                        hourly_chart,
                    ],
                    width=12,
                ),
                className="my-3",
            ),
        ]
        layout_children.extend(latest_strip_modals)
        return dbc.Container(layout_children, fluid=True)

    RUNTIME_BOOL_KEYS = {"DAY_AND_NIGHT_CAPTURE", "TELEGRAM_ENABLED"}
    RUNTIME_NUMBER_KEYS = {
        "CONFIDENCE_THRESHOLD_DETECTION",
        "SAVE_THRESHOLD",
        "MAX_FPS_DETECTION",
        "CLASSIFIER_CONFIDENCE_THRESHOLD",
        "FUSION_ALPHA",
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "TELEGRAM_COOLDOWN",
    }

    RUNTIME_KEYS_ORDER = [
        "CONFIDENCE_THRESHOLD_DETECTION",
        "SAVE_THRESHOLD",
        "MAX_FPS_DETECTION",
        "CLASSIFIER_CONFIDENCE_THRESHOLD",
        "FUSION_ALPHA",
        "DAY_AND_NIGHT_CAPTURE",
        "DAY_AND_NIGHT_CAPTURE_LOCATION",
        "STREAM_FPS",
        "STREAM_FPS_CAPTURE",
        "TELEGRAM_COOLDOWN",
        "TELEGRAM_ENABLED",
        "EDIT_PASSWORD",
    ]

    SYSTEM_KEYS_ORDER = [
        "OUTPUT_DIR",
        "MODEL_BASE_PATH",
        "DEBUG_MODE",
        "CPU_LIMIT",
        "VIDEO_SOURCE",
        "DETECTOR_MODEL_CHOICE",
        "STREAM_WIDTH_OUTPUT_RESIZE",
        "LOCATION_DATA",
    ]

    def _format_value(value):
        if isinstance(value, dict) and "latitude" in value and "longitude" in value:
            return f"{value['latitude']}, {value['longitude']}"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def settings_layout():
        """Layout for the Settings page."""
        payload = get_settings_payload()

        runtime_rows = []
        for key in RUNTIME_KEYS_ORDER:
            if key not in payload:
                continue
            meta = payload[key]
            value = meta["value"]
            default = meta["default"]
            if key in RUNTIME_BOOL_KEYS:
                input_component = dbc.Select(
                    id={"type": "runtime-setting", "key": key},
                    options=[
                        {"label": "true", "value": True},
                        {"label": "false", "value": False},
                    ],
                    value=value,
                )
            elif key in RUNTIME_NUMBER_KEYS:
                input_component = dbc.Input(
                    id={"type": "runtime-setting", "key": key},
                    type="number",
                    value=value,
                )
            else:
                input_component = dbc.Input(
                    id={"type": "runtime-setting", "key": key},
                    type="text",
                    value=_format_value(value),
                )
            runtime_rows.append(
                html.Tr(
                    [
                        html.Td(key),
                        html.Td(input_component),
                        html.Td(_format_value(default)),
                        html.Td(meta["source"]),
                        html.Td("yes" if meta["restart_required"] else "no"),
                    ]
                )
            )

        system_rows = []
        for key in SYSTEM_KEYS_ORDER:
            if key not in payload:
                continue
            meta = payload[key]
            system_rows.append(
                html.Tr(
                    [
                        html.Td(key),
                        html.Td(_format_value(meta["value"])),
                        html.Td(_format_value(meta["default"])),
                        html.Td(meta["source"]),
                        html.Td("yes" if meta["restart_required"] else "no"),
                    ]
                )
            )

        return dbc.Container(
            [
                generate_navbar(),
                html.H2("Settings", className="text-center my-3"),
                html.H4("Runtime Settings", className="mt-4"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Key"),
                                    html.Th("Value"),
                                    html.Th("Default"),
                                    html.Th("Source"),
                                    html.Th("Restart"),
                                ]
                            )
                        ),
                        html.Tbody(runtime_rows),
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                ),
                dbc.Button("Apply", id="settings-apply", color="primary"),
                html.Div(id="settings-status", className="mt-3"),
                html.H4("System Settings (Read-only)", className="mt-5"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Key"),
                                    html.Th("Value"),
                                    html.Th("Default"),
                                    html.Th("Source"),
                                    html.Th("Restart"),
                                ]
                            )
                        ),
                        html.Tbody(system_rows),
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                ),
            ],
            fluid=True,
        )

    def gallery_layout():
        """Layout for the gallery page (calls generate_gallery which uses classes)."""
        return generate_gallery()

    # --- App Layout Modification ---
    initial_page_content = dbc.Container(
        [
            generate_navbar(),
            dcc.Loading(
                type="circle",
                children=html.Div(
                    html.P("Loading...", className="text-center my-5"),
                    className="d-flex justify-content-center",
                ),
            ),
        ],
        fluid=True,
    )
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(
                id="auth-status-store",
                storage_type="session",
                data={"authenticated": False},
            ),  # Stores auth flag
            dcc.Store(
                id="edit-target-date-store", storage_type="memory"
            ),  # Temp store for target date
            html.Div(id="page-content", children=initial_page_content),  # Main page content with initial shell
            html.Div(id="modal-container"),  # Container for the password modal
            # Other stores/hidden divs if needed
            dcc.Store(id="scroll-trigger-store", data=None),
            html.Div(id="dummy-clientside-output-div", style={"display": "none"}),
        ]
    )

    # -----------------------------
    # Dash Callbacks
    # -----------------------------
    @app.callback(
        Output("today-species-summary", "children"),
        Input("today-species-summary-trigger", "n_intervals"),
    )
    def load_today_species_summary(n_intervals):
        if n_intervals is None:
            raise PreventUpdate
        date_iso = datetime.now().strftime("%Y-%m-%d")
        summary = get_daily_species_summary(date_iso)
        if not summary:
            return html.P(
                "No species detected yet today.",
                className="text-center text-muted",
            )
        rows = []
        for item in summary[:12]:  # limit to keep render light
            display_name = item["common_name"] or item["species"]
            rows.append(
                html.Tr(
                    [
                        html.Td(display_name),
                        html.Td(item["species"]),
                        html.Td(item["count"]),
                    ]
                )
            )
        return dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Name"),
                            html.Th("Species ID"),
                            html.Th("Count"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            bordered=True,
            striped=True,
        hover=True,
        size="sm",
    )

    @app.callback(
        Output("today-daily-summary", "children"),
        Input("today-daily-summary-trigger", "n_intervals"),
    )
    def load_today_daily_summary(n_intervals):
        if n_intervals is None:
            raise PreventUpdate
        date_iso = datetime.now().strftime("%Y-%m-%d")
        try:
            return get_gallery_daily_summary(date_iso)
        except Exception as e:
            logger.error(f"Error rendering daily summary for landing ({date_iso}): {e}")
            return html.P(
                "Could not load daily summary.",
                className="text-center text-muted",
            )

    # --- ADD Password Modal Structure ---
    @app.callback(
        Output("modal-container", "children"),
        Input(
            "url", "pathname"
        ),  # Trigger whenever URL changes to ensure modal is added
    )
    def add_password_modal(_):  # We don't need the pathname here, just need to trigger
        return dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Password Required")),
                dbc.ModalBody(
                    [
                        dbc.Alert(
                            "Please enter the password to edit this page.",
                            color="info",
                            id="password-modal-message",
                        ),
                        dbc.Input(
                            id="password-input",
                            type="password",
                            placeholder="Enter password...",
                        ),
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Confirm",
                        id="submit-password-button",
                        color="primary",
                        n_clicks=0,
                    )
                ),
            ],
            id="password-modal",
            is_open=False,  # Initially closed
            backdrop="static",  # Prevent closing by clicking outside
            keyboard=True,  # Allow closing with Esc key (might want to disable if annoying)
        )

    # --- Callback to Open Password Modal ---
    @app.callback(
        Output("password-modal", "is_open", allow_duplicate=True),
        Output("edit-target-date-store", "data"),
        Output("password-modal-message", "children", allow_duplicate=True),
        Output("password-input", "value"),  # Clear password input
        Input({"type": "open-edit-modal-button", "date": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_password_modal(n_clicks):
        ctx = callback_context
        if (
            not ctx.triggered
            or not any(n_clicks)
            or all(c == 0 for c in n_clicks if c is not None)
        ):
            raise PreventUpdate

        # Get the date from the button that was clicked
        button_id = ctx.triggered_id
        if (
            isinstance(button_id, dict)
            and button_id.get("type") == "open-edit-modal-button"
        ):
            target_date = button_id.get("date")
            if target_date:
                # Reset message and open modal
                message = dbc.Alert(
                    "Please enter the password to edit this page.",
                    color="info",
                )
                return (
                    True,
                    {"date": target_date},
                    message,
                    "",
                )  # Open modal, store date, set message, clear input

        raise PreventUpdate

    # --- Callback to Check Password and Redirect (WITH DEBUG LOGGING) ---
    @app.callback(
        Output("password-modal", "is_open", allow_duplicate=True),
        Output("auth-status-store", "data"),
        Output("url", "pathname"),  # Output to trigger navigation
        Output(
            "password-modal-message", "children", allow_duplicate=True
        ),  # Show error message
        Input("submit-password-button", "n_clicks"),
        State("password-input", "value"),
        State("edit-target-date-store", "data"),
        State("auth-status-store", "data"),  # Get current auth status
        prevent_initial_call=True,
    )
    def check_password(n_clicks, entered_password, target_date_data, auth_data):
        if n_clicks == 0 or n_clicks is None:
            logger.warning(
                "check_password: Callback triggered but n_clicks is 0 or None."
            )
            raise PreventUpdate  # No actual click submission

        if not target_date_data:
            logger.error("check_password: Target date data is missing from store.")
            # Provide feedback in modal?
            return (
                True,
                no_update,
                no_update,
                dbc.Alert("Error: Target date not found.", color="danger"),
            )

        target_date = target_date_data.get("date")
        if not target_date:
            logger.error("check_password: Target date missing within data.")
            return (
                True,
                no_update,
                no_update,
                dbc.Alert("Error: Target date invalid.", color="danger"),
            )

        # Check if password was entered
        if not entered_password:
            logger.warning("check_password: No password entered by user.")
            return (
                True,
                no_update,
                no_update,
                dbc.Alert("Please enter a password.", color="warning"),
            )

        # --- Perform the comparison ---
        # Use .strip() to handle potential accidental whitespace
        password_match = False
        if EDIT_PASSWORD and entered_password:
            password_match = entered_password.strip() == EDIT_PASSWORD.strip()

        if password_match:
            # Correct password
            logger.info(
                f"Password correct for editing date: {target_date}. Redirecting..."
            )
            new_auth_data = {"authenticated": True}
            redirect_path = f"/edit/{target_date}"
            # Close modal, update auth store, redirect, reset message (no_update)
            return False, new_auth_data, redirect_path, no_update
        else:
            # Incorrect password
            logger.warning(
                f"Incorrect password entered for editing date: {target_date}"
            )
            error_message = dbc.Alert("Incorrect password!", color="danger")
            # Keep modal open, don't change auth store, don't redirect, show error
            return True, no_update, no_update, error_message

    @app.callback(
        Output({"type": "daily-fused-agreement-modal", "index": ALL}, "is_open"),
        [
            Input(
                {"type": "daily-fused-agreement-thumbnail", "index": ALL}, "n_clicks"
            ),
            Input({"type": "daily-fused-agreement-close", "index": ALL}, "n_clicks"),
            Input(
                {"type": "daily-fused-agreement-modal-image", "index": ALL}, "n_clicks"
            ),
        ],
        [State({"type": "daily-fused-agreement-modal", "index": ALL}, "is_open")],
    )
    def toggle_daily_fused_agreement_modal(
        thumbnail_clicks, close_clicks, modal_image_clicks, current_states
    ):
        return _toggle_modal_generic(
            open_trigger_type="daily-fused-agreement-thumbnail",
            close_button_trigger_type="daily-fused-agreement-close",
            close_image_trigger_type="daily-fused-agreement-modal-image",
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states,
        )

    @app.callback(
        Output({"type": "daily-fused-weighted-modal", "index": ALL}, "is_open"),
        [
            Input({"type": "daily-fused-weighted-thumbnail", "index": ALL}, "n_clicks"),
            Input({"type": "daily-fused-weighted-close", "index": ALL}, "n_clicks"),
            Input(
                {"type": "daily-fused-weighted-modal-image", "index": ALL}, "n_clicks"
            ),
        ],
        [State({"type": "daily-fused-weighted-modal", "index": ALL}, "is_open")],
    )
    def toggle_daily_fused_weighted_modal(
        thumbnail_clicks, close_clicks, modal_image_clicks, current_states
    ):
        return _toggle_modal_generic(
            open_trigger_type="daily-fused-weighted-thumbnail",
            close_button_trigger_type="daily-fused-weighted-close",
            close_image_trigger_type="daily-fused-weighted-modal-image",
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states,
        )

    @app.callback(
        Output({"type": "alltime-fused-agreement-modal", "index": ALL}, "is_open"),
        [
            Input(
                {"type": "alltime-fused-agreement-thumbnail", "index": ALL}, "n_clicks"
            ),
            Input({"type": "alltime-fused-agreement-close", "index": ALL}, "n_clicks"),
            Input(
                {"type": "alltime-fused-agreement-modal-image", "index": ALL},
                "n_clicks",
            ),
        ],
        [State({"type": "alltime-fused-agreement-modal", "index": ALL}, "is_open")],
    )
    def toggle_alltime_fused_agreement_modal(
        thumbnail_clicks, close_clicks, modal_image_clicks, current_states
    ):
        return _toggle_modal_generic(
            open_trigger_type="alltime-fused-agreement-thumbnail",
            close_button_trigger_type="alltime-fused-agreement-close",
            close_image_trigger_type="alltime-fused-agreement-modal-image",
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states,
        )

    @app.callback(
        Output({"type": "alltime-fused-weighted-modal", "index": ALL}, "is_open"),
        [
            Input(
                {"type": "alltime-fused-weighted-thumbnail", "index": ALL}, "n_clicks"
            ),
            Input({"type": "alltime-fused-weighted-close", "index": ALL}, "n_clicks"),
            Input(
                {"type": "alltime-fused-weighted-modal-image", "index": ALL}, "n_clicks"
            ),
        ],
        [State({"type": "alltime-fused-weighted-modal", "index": ALL}, "is_open")],
    )
    def toggle_alltime_fused_weighted_modal(
        thumbnail_clicks, close_clicks, modal_image_clicks, current_states
    ):
        return _toggle_modal_generic(
            open_trigger_type="alltime-fused-weighted-thumbnail",
            close_button_trigger_type="alltime-fused-weighted-close",
            close_image_trigger_type="alltime-fused-weighted-modal-image",
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states,
        )

    @app.callback(
        Output({"type": "alltime-detector-modal", "index": ALL}, "is_open"),
        [
            Input({"type": "alltime-detector-thumbnail", "index": ALL}, "n_clicks"),
            Input({"type": "alltime-detector-close", "index": ALL}, "n_clicks"),
            Input({"type": "alltime-detector-modal-image", "index": ALL}, "n_clicks"),
        ],
        [State({"type": "alltime-detector-modal", "index": ALL}, "is_open")],
    )
    def toggle_alltime_detector_modal(
        thumbnail_clicks, close_clicks, modal_image_clicks, current_states
    ):
        return _toggle_modal_generic(
            open_trigger_type="alltime-detector-thumbnail",
            close_button_trigger_type="alltime-detector-close",
            close_image_trigger_type="alltime-detector-modal-image",
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states,
        )

    @app.callback(
        Output({"type": "alltime-classifier-modal", "index": ALL}, "is_open"),
        [
            Input({"type": "alltime-classifier-thumbnail", "index": ALL}, "n_clicks"),
            Input({"type": "alltime-classifier-close", "index": ALL}, "n_clicks"),
            Input({"type": "alltime-classifier-modal-image", "index": ALL}, "n_clicks"),
        ],
        [State({"type": "alltime-classifier-modal", "index": ALL}, "is_open")],
    )
    def toggle_alltime_classifier_modal(
        thumbnail_clicks, close_clicks, modal_image_clicks, current_states
    ):
        return _toggle_modal_generic(
            open_trigger_type="alltime-classifier-thumbnail",
            close_button_trigger_type="alltime-classifier-close",
            close_image_trigger_type="alltime-classifier-modal-image",
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states,
        )

    @app.callback(
        Output({"type": "latest-strip-modal", "index": ALL}, "is_open"),
        [
            Input({"type": "latest-strip-thumbnail", "index": ALL}, "n_clicks"),
            Input({"type": "latest-strip-close", "index": ALL}, "n_clicks"),
            Input({"type": "latest-strip-modal-image", "index": ALL}, "n_clicks"),
        ],
        [State({"type": "latest-strip-modal", "index": ALL}, "is_open")],
    )
    def toggle_latest_strip_modal(
        thumbnail_clicks, close_clicks, modal_image_clicks, current_states
    ):
        return _toggle_modal_generic(
            open_trigger_type="latest-strip-thumbnail",
            close_button_trigger_type="latest-strip-close",
            close_image_trigger_type="latest-strip-modal-image",
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states,
        )

    # --- Callback to Display Page ---
    @app.callback(
        Output("page-content", "children"),
        Output("scroll-trigger-store", "data"),
        Input("url", "pathname"),
        Input("url", "search"),
        State("auth-status-store", "data"),  # <-- ADD Auth Status State
    )
    def display_page(pathname, search, auth_data):
        scroll_trigger = no_update
        ctx = callback_context
        today_date_iso = datetime.now().strftime("%Y-%m-%d")

        is_subgallery_page_nav = (
            pathname is not None
            and pathname.startswith("/gallery/")
            and ctx.triggered
            and ctx.triggered[0]["prop_id"] == "url.search"
        )
        if is_subgallery_page_nav:
            scroll_trigger = time.time()

        if pathname is not None and pathname.startswith("/edit/"):
            date_str_iso = pathname.split("/")[-1]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str_iso):
                return "Invalid date format.", no_update

            # CHECK 1: Authentication
            if not auth_data or not auth_data.get("authenticated"):
                logger.warning(f"Unauthorized attempt to access edit page: {pathname}")
                return (
                    dbc.Container(
                        [  # Simple access denied page
                            generate_navbar(),
                            html.H2(
                                "Access Denied",
                                className="text-danger text-center mt-4",
                            ),
                            html.P(
                                "You must be authenticated to view this page.",
                                className="text-center",
                            ),
                            dbc.Button(
                                "Back to Gallery", href="/gallery", color="primary"
                            ),
                        ],
                        fluid=True,
                    ),
                    no_update,
                )

            # CHECK 2: Prevent editing today's data
            if date_str_iso == today_date_iso:
                logger.warning(
                    f"Authenticated user attempted to access edit page for current day: {date_str_iso}"
                )
                return (
                    dbc.Container(
                        [
                            generate_navbar(),
                            html.H2(
                                f"Edit Images from {date_str_iso}",
                                className="text-center my-3",
                            ),
                            dbc.Alert(
                                "Editing the gallery for the current day is not allowed.",
                                color="warning",
                            ),
                            dbc.Button(
                                "Back to Subgallery",
                                href=f"/gallery/{date_str_iso}",
                                color="secondary",
                                className="me-2",
                            ),
                            dbc.Button(
                                "Back to Main Gallery",
                                href="/gallery",
                                color="secondary",
                            ),
                        ],
                        fluid=True,
                    ),
                    no_update,
                )

            # If authenticated AND not today, generate the edit page
            logger.info(f"Authorized access to edit page: {pathname}")
            return generate_edit_page(date_str_iso), no_update

        # --- SUBGALLERY ROUTE ---
        elif pathname is not None and pathname.startswith("/gallery/"):
            date = pathname.split("/")[-1]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                return "Invalid date format.", no_update
            page = 1
            if search:
                params = parse_qs(search.lstrip("?"))
                if "page" in params:
                    try:
                        page = int(params["page"][0])
                    except ValueError:
                        page = 1
            content = generate_subgallery(date, page)
            return content, scroll_trigger
        # --- OTHER ROUTES ---
        elif pathname == "/gallery":
            return generate_gallery(), no_update
        elif pathname == "/species":
            return species_summary_layout(), no_update
        elif pathname == "/settings":
            return settings_layout(), no_update
        elif pathname == "/" or pathname is None or pathname == "":
            return stream_layout(), no_update
        # --- 404 ---
        else:
            logger.warning(f"404 Not Found for pathname: {pathname}")
            return (
                dbc.Container(
                    [
                        generate_navbar(),
                        html.H1(
                            "404 - Page Not Found",
                            className="text-center text-danger mt-5",
                        ),
                        html.P(
                            f"The path '{pathname}' was not recognized.",
                            className="text-center",
                        ),
                        dbc.Button("Go to Homepage", href="/", color="primary"),
                    ],
                    fluid=True,
                ),
                no_update,
            )

    # Callback to update the selected images store based on checkboxes
    @app.callback(
        Output("settings-status", "children"),
        Input("settings-apply", "n_clicks"),
        State({"type": "runtime-setting", "key": ALL}, "value"),
        State({"type": "runtime-setting", "key": ALL}, "id"),
        prevent_initial_call=True,
    )
    def apply_runtime_settings(n_clicks, values, ids):
        if not n_clicks:
            raise PreventUpdate
        updates = {}
        for value, meta in zip(values, ids):
            updates[meta["key"]] = value
        valid, errors = validate_runtime_updates(updates)
        if errors:
            messages = [f"{key}: {msg}" for key, msg in errors.items()]
            return dbc.Alert(
                html.Ul([html.Li(msg) for msg in messages]),
                color="danger",
                dismissable=True,
            )
        update_runtime_settings(valid)
        return dbc.Alert(
            "Runtime settings updated. Changes apply immediately where supported.",
            color="success",
            dismissable=True,
        )

    @app.callback(
        Output("selected-images-store", "data"),
        Output("selected-count", "children"),
        Output("selected-summary-count", "children"),
        Output("selected-summary-list", "children"),
        Output("selected-summary-count-bottom", "children"),
        Output("selected-summary-list-bottom", "children"),
        Input({"type": "edit-image-checkbox", "index": ALL}, "value"),
        State({"type": "edit-image-checkbox", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def update_selected_images(checkbox_values, checkbox_ids):
        selected_paths = []
        if not checkbox_ids:
            return (
                [],
                "Selected: 0",
                "0",
                "None",
                "0",
                "None",
            )
        for i, cb_id in enumerate(checkbox_ids):
            is_checked = checkbox_values[i] if i < len(checkbox_values) else False
            if is_checked:
                relative_path = cb_id["index"]
                selected_paths.append(relative_path)
        count_text = f"Selected: {len(selected_paths)}"
        short_list = (
            ", ".join([os.path.basename(p) for p in selected_paths[:3]])
            if selected_paths
            else "None"
        )
        if len(selected_paths) > 3:
            short_list += ", …"
        return (
            selected_paths,
            count_text,
            str(len(selected_paths)),
            short_list,
            str(len(selected_paths)),
            short_list,
        )

    @app.callback(
        Output("edit-gallery-grid", "children", allow_duplicate=True),
        Output("selected-images-store", "data", allow_duplicate=True),
        Output("selected-count", "children", allow_duplicate=True),
        Output("selected-summary-count", "children", allow_duplicate=True),
        Output("selected-summary-list", "children", allow_duplicate=True),
        Output("selected-summary-count-bottom", "children", allow_duplicate=True),
        Output("selected-summary-list-bottom", "children", allow_duplicate=True),
        Input("edit-filter-apply", "n_clicks"),
        State("edit-filter-status", "value"),
        State("edit-sort", "value"),
        State("edit-filter-detector-min", "value"),
        State("edit-filter-classifier-min", "value"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def apply_edit_filters(n_clicks, filter_status, sort_value, det_min, cls_min, pathname):
        match = re.search(r"/edit/(\d{4}-\d{2}-\d{2})", pathname or "")
        if not match:
            raise PreventUpdate
        date_str_iso = match.group(1)
        df = read_csv_for_date(date_str_iso)
        if df.empty:
            return (
                [],
                [],
                "Selected: 0",
                "0",
                "None",
                "0",
                "None",
            )
        # Filter
        if filter_status == "downloaded":
            df = df[df["downloaded_timestamp"].astype(str).str.strip() != ""]
        elif filter_status == "not_downloaded":
            df = df[df["downloaded_timestamp"].astype(str).str.strip() == ""]
        # Threshold filters
        try:
            if det_min is not None:
                df = df[pd.to_numeric(df["best_class_conf"], errors="coerce") >= float(det_min)]
        except Exception:
            pass
        try:
            if cls_min is not None:
                df = df[pd.to_numeric(df["top1_confidence"], errors="coerce") >= float(cls_min)]
        except Exception:
            pass
        # Sort
        if sort_value == "time_asc":
            df = df.sort_values(by="timestamp", ascending=True)
        elif sort_value == "time_desc":
            df = df.sort_values(by="timestamp", ascending=False)
        elif sort_value == "species":
            df = df.sort_values(
                by=["best_class", "top1_class_name", "timestamp"], ascending=[True, True, False]
            )
        tiles = build_edit_tiles(date_str_iso, df)
        return tiles, [], "Selected: 0", "0", "None", "0", "None"

    @app.callback(
        Output({"type": "edit-image-checkbox", "index": ALL}, "value"),
        Input("select-all-button", "n_clicks"),
        Input("clear-selection-button", "n_clicks"),
        Input("edit-filter-apply", "n_clicks"),
        State({"type": "edit-image-checkbox", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def bulk_select_clear(select_all_clicks, clear_clicks, filter_clicks, current_values):
        current_values = current_values or []
        ctx = callback_context
        if not ctx.triggered or not current_values:
            raise PreventUpdate
        trigger_id = ctx.triggered_id
        if trigger_id == "select-all-button":
            return [True for _ in current_values]
        if trigger_id == "clear-selection-button":
            return [False for _ in current_values]
        # filter apply should not change selection
        raise PreventUpdate

    # Callback to trigger delete confirmation with preview
    @app.callback(
        Output("confirm-delete", "displayed"),
        Output("confirm-delete", "message"),
        Input("delete-button", "n_clicks"),
        Input("delete-button-bottom", "n_clicks"),  # Trigger from bottom button too
        State("selected-images-store", "data"),
        prevent_initial_call=True,
    )
    def display_delete_confirm(n_clicks_top, n_clicks_bottom, selected_images):
        triggered = (n_clicks_top and n_clicks_top > 0) or (
            n_clicks_bottom and n_clicks_bottom > 0
        )
        if not triggered:
            raise PreventUpdate
        count = len(selected_images or [])
        if count == 0:
            return False, "No images selected."
        preview = ", ".join([os.path.basename(p) for p in (selected_images or [])][:5])
        if count > 5:
            preview += ", …"
        msg = (
            f"Delete {count} image(s)? This permanently removes files and database rows. "
            f"Selection: {preview}"
        )
        return True, msg

    # Callback to handle deletion after confirmation
    @app.callback(
        Output("edit-status-message", "children", allow_duplicate=True),
        Output("edit-gallery-grid", "children", allow_duplicate=True),
        Output("selected-images-store", "data", allow_duplicate=True),
        Output("selected-count", "children", allow_duplicate=True),
        Output("selected-summary-count", "children", allow_duplicate=True),
        Output("selected-summary-list", "children", allow_duplicate=True),
        Output("selected-summary-count-bottom", "children", allow_duplicate=True),
        Output("selected-summary-list-bottom", "children", allow_duplicate=True),
        Input("confirm-delete", "submit_n_clicks"),
        State("selected-images-store", "data"),
        State("url", "pathname"),  # Get the date from the current URL
        prevent_initial_call=True,
    )
    def handle_delete(submit_n_clicks, selected_images, pathname):
        if not submit_n_clicks or submit_n_clicks == 0:
            raise PreventUpdate  # No submission yet

        if not selected_images:
            return (
                dbc.Alert("No images selected for deletion.", color="warning"),
                no_update,
                [],
                "Selected: 0",
                "0",
                "None",
                "0",
                "None",
            )  # Or just no_update

        # Extract date from pathname like /edit/YYYY-MM-DD
        match = re.search(r"/edit/(\d{4}-\d{2}-\d{2})", pathname)
        if not match:
            return (
                dbc.Alert("Error: Could not determine date from URL.", color="danger"),
                no_update,
            )
        date_str_iso = match.group(1)

        df = read_csv_for_date(date_str_iso)
        if df.empty:
            return (
                dbc.Alert(
                    f"Error: Could not read data for {date_str_iso} to perform deletion.",
                    color="danger",
                ),
                no_update,
                [],
                "Selected: 0",
                "0",
                "None",
                "0",
                "None",
            )

        selected_filenames = {os.path.basename(p) for p in selected_images}
        try:
            delete_images_by_names(db_conn, selected_filenames)
            success_delete = True
            _cached_images["images"] = None
            _cached_images["timestamp"] = 0
        except Exception as e:
            logger.error(f"Error deleting rows from SQLite: {e}")
            success_delete = False

        deleted_files_count = 0
        error_messages = []

        # Delete image files
        for relative_path in selected_images:
            if not delete_image_files(relative_path):
                error_messages.append(
                    f"Failed to delete one or more files for {os.path.basename(relative_path)}"
                )

        # --- Feedback Message ---
        status_messages = []
        if success_delete:
            status_messages.append(f"Successfully removed {len(selected_images)} entries from database.")
        else:
            status_messages.append(f"Error updating database for {date_str_iso}.")

        if deleted_files_count > 0 or not error_messages:
            status_messages.append(
                f"Attempted deletion of files for {len(selected_images)} entries."
            )  # Be slightly vague if some failed
        if error_messages:
            status_messages.extend(error_messages)

        alert_color = (
            "success"
            if success_delete and not error_messages
            else ("warning" if success_delete else "danger")
        )

        # Rebuild gallery after deletion to reflect current state
        refreshed_df = read_csv_for_date(date_str_iso)
        refreshed_tiles = build_edit_tiles(date_str_iso, refreshed_df) if not refreshed_df.empty else []

        return (
            dbc.Alert(
                html.Ul([html.Li(msg) for msg in status_messages]),
                color=alert_color,
                dismissable=True,
            ),
            refreshed_tiles,
            [],
            "Selected: 0",
            "0",
            "None",
            "0",
            "None",
        )

    # Callback to handle download request
    @app.callback(
        Output("download-zip", "data"),
        Output(
            "edit-status-message", "children", allow_duplicate=True
        ),  # Update status too
        Input("download-button", "n_clicks"),
        Input("download-button-bottom", "n_clicks"),
        State("selected-images-store", "data"),
        State("url", "pathname"),
        prevent_initial_call=True,
    )
    def handle_download(n_clicks_top, n_clicks_bottom, selected_images, pathname):
        triggered = (
            callback_context.triggered_id == "download-button"
            or callback_context.triggered_id == "download-button-bottom"
        )
        if not triggered:
            raise PreventUpdate

        if not selected_images:
            return no_update, dbc.Alert(
                "No images selected for download.", color="warning", dismissable=True
            )

        # Extract date from pathname
        match = re.search(r"/edit/(\d{4}-\d{2}-\d{2})", pathname)
        if not match:
            return no_update, dbc.Alert(
                "Error: Could not determine date from URL.",
                color="danger",
                dismissable=True,
            )
        date_str_iso = match.group(1)

        df = read_csv_for_date(date_str_iso)
        if df.empty:
            return no_update, dbc.Alert(
                f"Error: Could not read data for {date_str_iso} to perform download.",
                color="danger",
                dismissable=True,
            )

        # --- 1. Update DB ---
        download_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        selected_filenames = {os.path.basename(p) for p in selected_images}

        try:
            update_downloaded_timestamp(
                db_conn, selected_filenames, download_timestamp
            )
            success_db = True
            _cached_images["images"] = None
            _cached_images["timestamp"] = 0
        except Exception as e:
            logger.error(f"Error updating downloaded_timestamp in SQLite: {e}")
            success_db = False

        # --- 2. Create Zip ---
        zip_buffer = io.BytesIO()
        errors_zipping = []
        files_added = 0
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for relative_path in selected_images:
                # We want the ORIGINAL file for download
                original_relative_path = relative_path.replace(
                    "_optimized", "_original"
                )
                absolute_original_path = os.path.join(
                    output_dir, original_relative_path
                )
                if os.path.exists(absolute_original_path):
                    try:
                        # Add file to zip, using the original filename as the archive name
                        zip_file.write(
                            absolute_original_path,
                            arcname=os.path.basename(original_relative_path),
                        )
                        files_added += 1
                    except Exception as e:
                        logger.error(
                            f"Error adding {absolute_original_path} to zip: {e}"
                        )
                        errors_zipping.append(os.path.basename(original_relative_path))
                else:
                    logger.warning(
                        f"Original file not found for download: {absolute_original_path}"
                    )
                    errors_zipping.append(
                        os.path.basename(original_relative_path) + " (Not Found)"
                    )

        zip_buffer.seek(0)

        # --- 3. Prepare Download Data ---
        zip_data = base64.b64encode(zip_buffer.read()).decode("utf-8")
        download_filename = f"watchmybirds_{date_str_iso.replace('-','')}_download.zip"
        download_dict = dict(
            content=zip_data,
            filename=download_filename,
            base64=True,
            type="application/zip",
        )

        # --- 4. Prepare Status Message ---
        status_messages = []
        if success_db:
            status_messages.append(
                f"Marked {len(selected_images)} images as downloaded for {date_str_iso}."
            )
        else:
            status_messages.append(f"Error updating database for {date_str_iso}.")

        if files_added > 0:
            status_messages.append(
                f"Prepared {files_added} original images for download."
            )
        if errors_zipping:
            status_messages.append(
                f"Could not add {len(errors_zipping)} files to the zip: {', '.join(errors_zipping[:3])}{'...' if len(errors_zipping)>3 else ''}"
            )

        alert_color = (
            "success"
            if success_db and files_added > 0 and not errors_zipping
            else "warning"
        )

        # Send download data and status message
        return download_dict, dbc.Alert(
            html.Ul([html.Li(msg) for msg in status_messages]),
            color=alert_color,
            dismissable=True,
        )

    def _toggle_modal_generic(
        # Pass the specific types expected for this modal group
        open_trigger_type: str,
        close_button_trigger_type: str,
        close_image_trigger_type: str,
        thumbnail_clicks,
        close_clicks,
        modal_image_clicks,
        current_states,
    ):
        """Generic function to toggle modals based on explicit trigger types."""
        ctx = callback_context
        # Initialize variables used in the try block
        triggered_prop = None
        triggered_value = None
        triggered_id_dict = None
        triggered_type = None
        triggered_component_index = None

        # Guard clause: No trigger or no outputs defined
        if not ctx.triggered or not ctx.outputs_list:
            # Returning no_update for all outputs is often preferred over raising PreventUpdate
            # when dealing with ALL pattern callbacks if some outputs might exist.
            num_outputs = (
                len(ctx.outputs_list)
                if ctx.outputs_list
                else (len(current_states) if current_states else 0)
            )
            if num_outputs > 0:
                return [no_update] * num_outputs
            else:
                # If truly nothing to update, PreventUpdate might be applicable,
                # but returning an empty list or handling upstream might be safer.
                # For now, let's stick to no_update based on state length if possible.
                if current_states is not None:
                    return [no_update] * len(current_states)
                return (
                    []
                )  # Or potentially raise PreventUpdate if appropriate for your broader app structure

        triggered_prop = ctx.triggered[0]["prop_id"]
        triggered_value = ctx.triggered[0]["value"]

        # Guard clause: Trigger value indicates no actual click/event
        if triggered_value is None or triggered_value == 0:
            return [no_update] * len(ctx.outputs_list)

        # Safely parse the trigger ID
        try:
            triggered_id_str = triggered_prop.split(".")[0]
            triggered_id_dict = json.loads(triggered_id_str)
            triggered_type = triggered_id_dict.get("type")
            triggered_component_index = triggered_id_dict.get("index")
        except (IndexError, json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.error(
                f"Error parsing trigger ID: {triggered_prop} -> {e}", exc_info=True
            )
            return [no_update] * len(ctx.outputs_list)

        # Guard clause: Parsed ID lacks necessary parts
        if triggered_type is None or triggered_component_index is None:
            logger.error(
                f"Invalid trigger type/index after parsing: {triggered_id_dict}"
            )
            return [no_update] * len(ctx.outputs_list)

        output_component_ids = [output["id"] for output in ctx.outputs_list]
        new_states = [no_update] * len(output_component_ids)

        # Find the index in the output list corresponding to the triggered component's index
        target_list_index = -1
        for i, output_id in enumerate(output_component_ids):
            if (
                isinstance(output_id, dict)
                and output_id.get("index") == triggered_component_index
            ):
                target_list_index = i
                break

        # Guard clause: Could not find the target modal in the output list
        if target_list_index == -1:
            logger.error(
                f"Could not find target modal index for trigger component index {triggered_component_index}"
            )
            return new_states  # Return no_update for all

        # Determine the expected type of the output modal component
        try:
            expected_modal_output_type = ctx.outputs_list[0]["id"]["type"]
        except (IndexError, KeyError, TypeError) as e:
            logger.error(
                f"Could not determine expected output type from context: {e}",
                exc_info=True,
            )
            return [no_update] * len(ctx.outputs_list)

        # Guard clause: Ensure state list matches output list
        if current_states is None or len(current_states) != len(output_component_ids):
            logger.error(
                f"Mismatch or missing current_states. Len states: {len(current_states) if current_states else 'None'}, Len outputs: {len(output_component_ids)}"
            )
            return [no_update] * len(output_component_ids)

        # --- Main Logic ---
        target_output_id = output_component_ids[target_list_index]

        # Check if the target output component matches the expected type for this modal group
        if (
            not isinstance(target_output_id, dict)
            or target_output_id.get("type") != expected_modal_output_type
        ):
            logger.error(
                f"Output type mismatch at target index {target_list_index}. Expected '{expected_modal_output_type}', Found '{target_output_id.get('type')}'"
            )
            return [no_update] * len(output_component_ids)  # Prevent update on mismatch

        is_currently_open = current_states[target_list_index]

        # --- Logic for Opening ---
        if triggered_type == open_trigger_type:
            if not is_currently_open:
                # Explicitly create the list of desired states: close all, open target
                final_states = [False] * len(output_component_ids)
                final_states[target_list_index] = True
                new_states = final_states
            # else: modal already open, do nothing (implicit no_update)

        # --- Logic for Closing ---
        elif (
            triggered_type == close_button_trigger_type
            or triggered_type == close_image_trigger_type
        ):
            if is_currently_open:  # Only close if it's open
                # Explicitly create the list: keep existing states, close target
                final_states = list(
                    current_states
                )  # Important: work from current state
                final_states[target_list_index] = False
                new_states = final_states
            # else: modal already closed, do nothing (implicit no_update)

        # else: trigger type didn't match open/close types, do nothing (implicit no_update)

        return new_states

    @app.callback(
        Output({"type": "subgallery-modal-modal", "index": ALL}, "is_open"),
        [
            Input({"type": "subgallery-thumbnail", "index": ALL}, "n_clicks"),
            Input({"type": "subgallery-modal-close", "index": ALL}, "n_clicks"),
            Input({"type": "subgallery-modal-modal-image", "index": ALL}, "n_clicks"),
        ],
        [State({"type": "subgallery-modal-modal", "index": ALL}, "is_open")],
    )
    def toggle_subgallery_modal(
        thumbnail_clicks, close_clicks, modal_image_clicks, current_states
    ):
        return _toggle_modal_generic(
            open_trigger_type="subgallery-thumbnail",
            close_button_trigger_type="subgallery-modal-close",
            close_image_trigger_type="subgallery-modal-modal-image",
            thumbnail_clicks=thumbnail_clicks,
            close_clicks=close_clicks,
            modal_image_clicks=modal_image_clicks,
            current_states=current_states,
        )

    app.clientside_callback(
        ClientsideFunction(namespace="clientside", function_name="scrollToPagination"),
        # Change this Output:
        Output(
            "dummy-clientside-output-div", "children"
        ),  # Target the dummy div instead
        Input("scroll-trigger-store", "data"),
        prevent_initial_call=True,
    )

    # -----------------------------
    # Function to Start the Web Interface
    # -----------------------------
    def run(debug=False, host="0.0.0.0", port=8050):
        logger.info(f"Starting Dash server on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)

    return {"app": app, "server": server, "run": run}
