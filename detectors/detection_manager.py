# ------------------------------------------------------------------------------
# Detection Manager Module for Object Detection
# detectors/detection_manager.py
# ------------------------------------------------------------------------------

import time
from config import get_config

config = get_config()
import re
import threading
from datetime import datetime, timedelta, timezone
import cv2
import pytz
from astral import Observer
from astral.geocoder import database, lookup
from astral.sun import sun
import piexif
import piexif.helper
from detectors.detector import Detector
from detectors.classifier import ImageClassifier
from camera.video_capture import VideoCapture
from utils.telegram_notifier import send_telegram_message
from logging_config import get_logger
from PIL import Image
import os
import json
import hashlib
import queue
from utils.db import (
    get_connection,
    insert_image,
    fetch_day_images,
    fetch_daily_species_summary,
)
from utils.daylight import is_daytime
from utils.collage import (
    build_candidate,
    build_square_tile,
    has_black_border,
    load_image_rgb,
    parse_timestamp,
    select_collage_candidates,
    trim_dark_borders,
)

"""
This module defines the DetectionManager class, which orchestrates video frame acquisition,
object detection, image classification, and saving of results, including EXIF metadata.
"""
logger = get_logger(__name__)


# >>> Helper Functions >>>
def degrees_to_dms_rational(degrees_float):
    """
    Converts decimal degrees to DMS rational format for EXIF.

    Args:
        degrees_float (float): Decimal degree value.

    Returns:
        list: DMS rational format [(deg,1),(min,1),(sec*1000,1000)].
    """
    degrees_float = abs(degrees_float)
    degrees = int(degrees_float)
    minutes_float = (degrees_float - degrees) * 60
    minutes = int(minutes_float)
    seconds_float = (minutes_float - minutes) * 60
    # Use rational representation (numerator, denominator)
    # Ensure seconds are non-negative before int conversion if very close to zero
    seconds_int = max(0, int(seconds_float * 1000))
    return [(degrees, 1), (minutes, 1), (seconds_int, 1000)]


def add_exif_metadata(image_path, capture_time, location_config=None):
    """
    Adds DateTimeOriginal and optional GPS EXIF data to an image file.

    Args:
        image_path (str): Path to the saved JPEG image.
        capture_time (datetime): Datetime representing capture time.
        location_config (dict, optional): Dict with 'latitude' and 'longitude'.

    Returns:
        None
    """
    try:
        # 1. Format DateTime for EXIF
        exif_dt_str = capture_time.strftime("%Y:%m:%d %H:%M:%S")
        # 2. Prepare EXIF dictionary structure
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_dt_str
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_dt_str
        # 3. Add GPS Data (if available AND is a dictionary)
        # Check if location_config is a dictionary and has the required keys
        if (
            isinstance(location_config, dict)
            and "latitude" in location_config
            and "longitude" in location_config
        ):
            try:
                lat = float(location_config["latitude"])
                lon = float(location_config["longitude"])
                gps_latitude = degrees_to_dms_rational(lat)
                gps_longitude = degrees_to_dms_rational(lon)
                exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = (
                    "N" if lat >= 0 else "S"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = gps_latitude
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = (
                    "E" if lon >= 0 else "W"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = gps_longitude
                utc_now = datetime.now(timezone.utc)
                exif_dict["GPS"][piexif.GPSIFD.GPSDateStamp] = utc_now.strftime(
                    "%Y:%m:%d"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSTimeStamp] = [
                    (utc_now.hour, 1),
                    (utc_now.minute, 1),
                    (max(0, int(utc_now.second * 1000)), 1000),
                ]
            except (ValueError, TypeError) as gps_e:
                logger.warning(
                    f"EXIF Warning: Could not parse GPS data from location_config: {location_config}. Error: {gps_e}"
                )
        elif location_config is not None:
            # Log a warning if location_config exists but isn't the expected format
            logger.warning(
                f"EXIF Warning: location_config is not a valid dictionary with lat/lon. Skipping GPS. Value: {location_config}"
            )
        # 4. Dump EXIF data to bytes
        exif_bytes = piexif.dump(exif_dict)
        # 5. Insert EXIF data into the image file
        piexif.insert(exif_bytes, image_path)
        logger.debug(f"Successfully added EXIF data to {os.path.basename(image_path)}")
    except FileNotFoundError:
        logger.error(f"EXIF Error: Image file not found at {image_path}")
    except Exception as e:
        # Log the specific exception and traceback
        logger.error(
            f"EXIF Error: Failed to add EXIF data to {os.path.basename(image_path)}. Error: {e}",
            exc_info=True,
        )


class DetectionManager:
    """
    Orchestrates the entire detection and classification pipeline.

    This class manages video frame acquisition, object detection, image classification,
    and result handling. It operates using two main threads: one for continuously
    capturing frames from a video source and another for processing these frames.

    Key Responsibilities:
    - Initializes and manages `VideoCapture`, `Detector`, and `ImageClassifier`.
    - Runs a frame acquisition loop to get the latest video frame.
    - Runs a detection loop that:
        - Performs object detection on frames.
        - Checks for daylight hours to operate.
        - Skips processing for identical consecutive frames.
        - If a detection meets the confidence threshold:
            - Saves original, optimized, and a square-cropped (zoomed) image of the detection.
            - Adds EXIF metadata (timestamp, GPS) to saved images.
            - Runs the `ImageClassifier` on the zoomed crop.
            - Records detection and classification results in a daily CSV log.
            - Sends a Telegram notification with an image, subject to a cooldown period.
    - Provides thread-safe mechanisms for accessing frames and managing components.
    - Handles graceful startup and shutdown of all components and threads.
    """

    def __init__(self):
        """
        Initializes the DetectionManager instance.

        Sets up configuration, initializes the classifier, creates thread locks for safe
        multithreaded operations, and prepares shared state variables. It also creates
        the necessary output directories and sets up the frame acquisition and detection
        threads without starting them.
        """
        self.config = config
        self.model_choice = self.config["DETECTOR_MODEL_CHOICE"]
        self.video_source = self.config["VIDEO_SOURCE"]
        self.location_config = self.config.get("LOCATION_DATA")
        self.debug = self.config["DEBUG_MODE"]
        self.SAVE_RESOLUTION_ZOOMED = (
            512  # Resolution for zoomed images, may be used for reclassification.
        )

        # Initializes the classifier.
        self.classifier = ImageClassifier()
        self.classifier_model_id = getattr(self.classifier, "model_id", "")
        print("Classifier initialized.")

        # Locks for thread-safe operations.
        self.frame_lock = threading.Lock()  # Protects raw frame and timestamp.
        self.detector_lock = (
            threading.Lock()
        )  # Protects detector instance during reinitialization.
        self.telegram_lock = threading.Lock()

        # Shared state.
        self.latest_raw_frame = None  # Updated continuously by the frame updater.
        self.latest_raw_timestamp = 0  # Timestamp for the raw frame.
        self.latest_detection_time = 0
        self.previous_frame_hash = None
        self.consecutive_identical_frames = 0

        # Statistics and notifications.
        self.detection_occurred = False
        self.last_notification_time = time.time()
        self.detection_counter = 0
        self.detection_classes_agg = set()
        self.last_frame_received_time = time.time()
        self.last_detection_event_time = time.time()
        self.last_status_alert_check = 0.0
        self.camera_offline_alert_active = False
        self.image_missing_alert_active = False
        self.no_detection_alert_active = False
        self.daylight_no_detection_seconds = 0.0
        self.last_daylight_check_ts = time.time()
        self.last_daylight_state = False
        self.image_missing_seconds = 60
        self.no_detection_seconds = 5 * 60 * 60
        self.status_alert_check_interval = 60

        # Video capture and detector instances.
        self.video_capture = None
        self.detector_instance = None
        self.detector_model_id = ""
        self.processing_queue = queue.Queue(maxsize=1)
        self.db_conn = get_connection()

        # Set up output directory
        self.output_dir = self.config["OUTPUT_DIR"]
        os.makedirs(self.output_dir, exist_ok=True)

        self.telegram_rule = (
            str(self.config.get("TELEGRAM_RULE", "basic")).strip().lower()
        )
        if self.telegram_rule not in {"basic", "daily_summary"}:
            logger.warning(
                f"Unknown TELEGRAM_RULE '{self.telegram_rule}', falling back to 'basic'."
            )
            self.telegram_rule = "basic"
        self.telegram_timezone = self._resolve_telegram_timezone()
        self.last_daily_summary_date = None

        # For clean shutdown.
        self.stop_event = threading.Event()

        # Daylight cache
        self._daytime_cache = {
            "city": None,
            "value": True,
            "ts": 0.0,
        }
        self._daytime_ttl = 300  # seconds

        # Two threads: one for frame acquisition and one for detection.
        self.frame_thread = threading.Thread(
            target=self._frame_update_loop, daemon=True
        )
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )

    # >>> Daytime Check Function >>>
    def _is_daytime(self, city_name):
        """Ermittelt Tageslichtstatus fuer die Detektion."""
        return is_daytime(
            city_name, self._daytime_cache, self._daytime_ttl, self.telegram_timezone
        )

    def _resolve_telegram_timezone(self):
        tz_name = str(self.config.get("TELEGRAM_TIMEZONE", "")).strip()
        if tz_name:
            try:
                return pytz.timezone(tz_name)
            except Exception as e:
                logger.warning(f"Invalid TELEGRAM_TIMEZONE '{tz_name}': {e}")

        location = self.config.get("DAY_AND_NIGHT_CAPTURE_LOCATION")
        if location:
            try:
                city = lookup(location, database())
                return pytz.timezone(city.timezone)
            except Exception as e:
                logger.warning(
                    "Failed to resolve timezone from DAY_AND_NIGHT_CAPTURE_LOCATION "
                    f"'{location}': {e}"
                )
        return pytz.UTC

    def _get_daily_summary_schedule(self, now_local):
        """Berechnet Sendezeitpunkt fuer Tageszusammenfassung vor Sonnenuntergang."""
        observer = None
        location_name = str(
            self.config.get("DAY_AND_NIGHT_CAPTURE_LOCATION", "")
        ).strip()
        if location_name:
            try:
                if "," in location_name:
                    lat_str, lon_str = location_name.split(",", 1)
                    observer = Observer(
                        latitude=float(lat_str), longitude=float(lon_str)
                    )
                else:
                    city = lookup(location_name, database())
                    observer = city.observer
            except Exception as exc:
                logger.warning(
                    f"Failed to resolve city '{location_name}' for sunset schedule: {exc}"
                )

        if observer is None and isinstance(self.location_config, dict):
            try:
                latitude = float(self.location_config.get("latitude"))
                longitude = float(self.location_config.get("longitude"))
                observer = Observer(latitude=latitude, longitude=longitude)
            except Exception as exc:
                logger.warning(
                    "Failed to build observer from LOCATION_DATA for sunset schedule: "
                    f"{exc}"
                )

        if observer is None:
            logger.warning("No valid location found for sunset-based daily summary.")
            return None

        try:
            sun_times = sun(
                observer,
                date=now_local.date(),
                tzinfo=self.telegram_timezone,
            )
        except Exception as exc:
            logger.warning(f"Failed to calculate sun times for daily summary: {exc}")
            return None

        sunset = sun_times.get("sunset")
        if not sunset:
            logger.warning("Sunset time missing for daily summary schedule.")
            return None

        return sunset - timedelta(minutes=20)

    def _maybe_send_daily_summary(self):
        if self.telegram_rule != "daily_summary":
            return
        if not self.config.get("TELEGRAM_ENABLED", True):
            return
        if not self.telegram_timezone:
            return

        now_local = datetime.now(self.telegram_timezone)
        today_str = now_local.strftime("%Y-%m-%d")
        if self.last_daily_summary_date == today_str:
            return

        scheduled = self._get_daily_summary_schedule(now_local)
        if not scheduled:
            return
        if now_local < scheduled:
            return

        self._send_daily_summary(today_str)
        self.last_daily_summary_date = today_str

    def _maybe_send_status_alerts(self):
        if not self.config.get("TELEGRAM_ENABLED", True):
            return
        if self.stop_event.is_set():
            return
        if self.video_capture is None:
            return
        now = time.time()
        if now - self.last_status_alert_check < self.status_alert_check_interval:
            return
        self.last_status_alert_check = now

        camera_running = (
            self.video_capture is not None and self.video_capture.is_running()
        )
        if not camera_running:
            if not self.camera_offline_alert_active:
                send_telegram_message("Camera disconnected or stream stopped.")
                self.camera_offline_alert_active = True
        else:
            self.camera_offline_alert_active = False

        if camera_running:
            missing_for = now - self.last_frame_received_time
            if missing_for >= self.image_missing_seconds:
                if not self.image_missing_alert_active:
                    send_telegram_message(
                        "No images received from camera for over 60 seconds."
                    )
                    self.image_missing_alert_active = True
                    try:
                        self.video_capture.request_reinitialize(
                            reason="No frames received for alert window."
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to request camera reinitialization: %s", exc
                        )
            else:
                self.image_missing_alert_active = False

        location = str(self.config.get("DAY_AND_NIGHT_CAPTURE_LOCATION", "")).strip()
        is_day = self._is_daytime(location) if location else True
        if is_day:
            if not self.last_daylight_state:
                self.last_daylight_check_ts = now
            else:
                self.daylight_no_detection_seconds += now - self.last_daylight_check_ts
                self.last_daylight_check_ts = now
            self.last_daylight_state = True
            if self.daylight_no_detection_seconds >= self.no_detection_seconds:
                if not self.no_detection_alert_active:
                    logger.info(
                        "No-detection daylight alert triggered: accumulated=%.1fs, "
                        "threshold=%.1fs, last_daylight_state=%s, last_check_ts=%.0f",
                        self.daylight_no_detection_seconds,
                        self.no_detection_seconds,
                        self.last_daylight_state,
                        self.last_daylight_check_ts,
                    )
                    send_telegram_message(
                        "No detections for more than 5 hours during daylight."
                    )
                    self.no_detection_alert_active = True
            else:
                self.no_detection_alert_active = False
        else:
            if self.last_daylight_state:
                self.daylight_no_detection_seconds = 0.0
            self.last_daylight_state = False
            self.last_daylight_check_ts = now
            self.no_detection_alert_active = False

    def _format_daily_summary_message(self, date_str_iso, rows):
        if not rows:
            return "Птицы кормушку не посещали"

        total_count = sum(int(row["count"]) for row in rows if row["count"] is not None)
        species_count = sum(1 for row in rows if row["species"])
        lines = [
            f"Итоги дня ({date_str_iso})",
            f"Накормлено птиц: {total_count}",
            f"Птичьи личности: {species_count}",
        ]
        for row in rows:
            species = row["species"]
            count = row["count"]
            if not species:
                continue
            display_name = str(species).replace("_", " ")
            lines.append(f"- {display_name}: {int(count)}")
        return "\n".join(lines)

    def _select_collage_paths(self, items):
        if not items:
            return []
        selected = select_collage_candidates(items, self.config)
        return [item["path"] for item in selected]

    def _load_square_image(self, image_path, size_px):
        image = load_image_rgb(image_path)
        if image is None:
            logger.warning(f"Failed to open image '{image_path}'.")
            return None
        if bool(self.config.get("COLLAGE_AVOID_BLACK_BORDERS", True)):
            image = trim_dark_borders(image)
        return build_square_tile(image, size_px)

    def _build_collage(self, image_paths, output_path):
        if not image_paths:
            return None
        grid_map = {1: 1, 4: 2, 9: 3}
        grid = grid_map.get(len(image_paths))
        if not grid:
            return None
        cell_size = 320
        collage_size = (grid * cell_size, grid * cell_size)
        collage = Image.new("RGB", collage_size, color=(0, 0, 0))

        for idx, image_path in enumerate(image_paths):
            row = idx // grid
            col = idx % grid
            tile = self._load_square_image(image_path, cell_size)
            if tile is None:
                continue
            collage.paste(tile, (col * cell_size, row * cell_size))

        try:
            collage.save(output_path, "JPEG", quality=85)
        except Exception as e:
            logger.warning(f"Failed to save collage '{output_path}': {e}")
            return None
        return output_path

    def _collect_daily_species_images(self, date_str_iso):
        date_prefix = date_str_iso.replace("-", "")
        rows = fetch_day_images(self.db_conn, date_str_iso)
        images_by_species = {}
        for row in rows:
            species = row["top1_class_name"] or row["best_class"]
            if not species:
                continue
            zoomed_name = row["zoomed_name"] or ""
            optimized_name = row["optimized_name"] or ""
            zoomed_path = (
                os.path.join(self.output_dir, date_prefix, zoomed_name)
                if zoomed_name
                else ""
            )
            optimized_path = (
                os.path.join(self.output_dir, date_prefix, optimized_name)
                if optimized_name
                else ""
            )
            prefer_zoomed = bool(self.config.get("COLLAGE_USE_ZOOMED", True))
            avoid_black = bool(self.config.get("COLLAGE_AVOID_BLACK_BORDERS", True))
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
            timestamp_value = parse_timestamp(row["timestamp"] or "")
            candidate = build_candidate(
                selected_path,
                confidence,
                timestamp_value,
                self.config,
                image=selected_image,
            )
            if candidate is None:
                continue
            images_by_species.setdefault(species, []).append(candidate)
        return images_by_species

    def _send_daily_summary(self, date_str_iso):
        rows = fetch_daily_species_summary(self.db_conn, date_str_iso)
        message = self._format_daily_summary_message(date_str_iso, rows)
        send_telegram_message(message)

        if not rows:
            return

        images_by_species = self._collect_daily_species_images(date_str_iso)
        if not images_by_species:
            return

        collage_dir = os.path.join(self.output_dir, "telegram")
        os.makedirs(collage_dir, exist_ok=True)
        species_order = [
            row["species"]
            for row in rows
            if row["species"] in images_by_species
        ]
        for species in species_order:
            items = images_by_species.get(species, [])
            selected_paths = self._select_collage_paths(items)
            if not selected_paths:
                continue
            grid_size = len(selected_paths)
            safe_species = re.sub(r"[^A-Za-z0-9_-]+", "_", str(species))
            collage_name = f"{date_str_iso}_{safe_species}_{grid_size}.jpg"
            collage_path = os.path.join(collage_dir, collage_name)
            result_path = self._build_collage(selected_paths, collage_path)
            if not result_path:
                continue
            display_name = str(species).replace("_", " ")
            send_telegram_message(f"Вид: {display_name}", photo_path=result_path)

    # >>> Initialize Components >>>
    def _initialize_components(self):
        """
        Initializes video capture and detector components without blocking startup.
        Safe to call repeatedly; returns True when both components are ready.
        """
        if self.stop_event.is_set():
            return False

        if self.video_capture is None:
            try:
                self.video_capture = VideoCapture(
                    self.video_source, debug=self.debug, auto_start=False
                )
                self.video_capture.start()
                logger.info("VideoCapture initialized in DetectionManager.")
            except Exception as e:
                logger.error(f"Failed to initialize video capture: {e}")
                self.video_capture = None

        if self.detector_instance is None:
            try:
                self.detector_instance = Detector(
                    model_choice=self.model_choice, debug=self.debug
                )
                model_id = getattr(self.detector_instance, "model_id", "") or ""
                if not model_id and hasattr(self.detector_instance, "model_path"):
                    model_id = os.path.basename(self.detector_instance.model_path)
                self.detector_model_id = model_id
                logger.info("Detector initialized in DetectionManager.")
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}")
                self.detector_instance = None

        return self.video_capture is not None and self.detector_instance is not None

    # >>> Frame Update Loop >>>
    def _frame_update_loop(self):
        """
        Continuously updates the latest raw frame from VideoCapture.

        Args:
            None

        Returns:
            None
        """
        while not self.stop_event.is_set():
            # Waits until video_capture is initialized.
            if self.video_capture is None:
                logger.debug("Video capture not initialized yet. Waiting...")
                time.sleep(0.1)
                continue

            frame = self.video_capture.get_frame()
            if frame is not None:
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
                    self.latest_raw_timestamp = time.time()
                self.last_frame_received_time = time.time()
                if self.image_missing_alert_active:
                    self.image_missing_alert_active = False
            else:
                # If no new frame for more than 5 seconds, mark it as unavailable
                if time.time() - self.latest_raw_timestamp > 5:
                    with self.frame_lock:
                        if self.latest_raw_frame is not None:
                            logger.warning(
                                "No new frames received for over 5 seconds. "
                                "Clearing latest_raw_frame to trigger placeholder display."
                            )
                        self.latest_raw_frame = None
                logger.debug("No frame available from VideoCapture in frame updater.")
                time.sleep(0.1)

    # >>> Create Square Crop Function >>>
    def create_square_crop(self, image, bbox, margin_percent=0.2, pad_color=None):
        """
        Erstellt einen quadratischen Zuschnitt um das Objekt und gleicht Randbereiche aus.

        Args:
            image (np.ndarray): The source image.
            bbox (tuple): Bounding box as (x1, y1, x2, y2).
            margin_percent (float): Extra margin percentage to add around the bbox.
            pad_color (tuple): Farbe fuer Randbereiche, falls konstant gepolstert wird.

        Returns:
            np.ndarray: The square cropped image.
        """
        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        bbox_width = bx2 - bx1
        bbox_height = by2 - by1
        bbox_side = max(bbox_width, bbox_height)
        new_side = int(bbox_side * (1 + margin_percent))
        desired_x1 = int(cx - new_side / 2)
        desired_y1 = int(cy - new_side / 2)
        desired_x2 = desired_x1 + new_side
        desired_y2 = desired_y1 + new_side
        image_h, image_w = image.shape[:2]
        crop_x1 = max(0, desired_x1)
        crop_y1 = max(0, desired_y1)
        crop_x2 = min(image_w, desired_x2)
        crop_y2 = min(image_h, desired_y2)
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        pad_left = crop_x1 - desired_x1
        pad_top = crop_y1 - desired_y1
        pad_right = desired_x2 - crop_x2
        pad_bottom = desired_y2 - crop_y2
        if pad_color is None:
            border_type = cv2.BORDER_REFLECT_101
            border_value = 0
        else:
            border_type = cv2.BORDER_CONSTANT
            border_value = pad_color
        square_crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=border_type,
            value=border_value,
        )
        return square_crop

    # >>> Detection Loop Thread Function >>>
    def _detection_loop(self):
        """
        Continuously processes the latest frame for detection.
        This is decoupled from frame acquisition.

        Args:
            None

        Returns:
            None
        """
        logger.info("Detection loop (worker) started.")
        while not self.stop_event.is_set():
            self._maybe_send_daily_summary()
            if not self._initialize_components():
                logger.debug("Components not ready yet; retrying shortly.")
                time.sleep(1)
                continue
            raw_frame = None
            capture_time_precise = datetime.now()
            # Grabs the most recent frame.
            with self.frame_lock:
                if self.latest_raw_frame is not None:
                    raw_frame = self.latest_raw_frame.copy()
            if raw_frame is None:
                self._maybe_send_status_alerts()
                logger.debug("No raw frame available for detection. Sleeping briefly.")
                self.previous_frame_hash = None
                self.consecutive_identical_frames = 0
                time.sleep(0.1)
                continue

            # >>> Identical Frame Check >>>
            try:
                current_frame_hash = hashlib.md5(raw_frame.tobytes()).hexdigest()
                if (
                    self.previous_frame_hash is not None
                    and current_frame_hash == self.previous_frame_hash
                ):
                    self.consecutive_identical_frames += 1
                    logger.warning(
                        f"Identical frame detected. Consecutive count: {self.consecutive_identical_frames}."
                    )
                    self.previous_frame_hash = current_frame_hash
                    target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
                    time.sleep(max(0.01, target_duration))
                    continue
                else:
                    self.consecutive_identical_frames = 0
                    self.previous_frame_hash = current_frame_hash
            except Exception as hash_e:
                logger.error(f"Error during frame hashing: {hash_e}")
                self.previous_frame_hash = None
                self.consecutive_identical_frames = 0

            if (
                self.config["DAY_AND_NIGHT_CAPTURE"]
                and not self._is_daytime(self.config["DAY_AND_NIGHT_CAPTURE_LOCATION"])
            ):
                self._maybe_send_status_alerts()
                logger.info("Not enough light for detection. Sleeping for 60 seconds.")
                time.sleep(60)
                continue

            start_time = time.time()
            try:
                object_detected, original_frame, detection_info_list = (
                    self.detector_instance.detect_objects(
                        raw_frame,
                        confidence_threshold=self.config[
                            "CONFIDENCE_THRESHOLD_DETECTION"
                        ],
                        save_threshold=self.config["SAVE_THRESHOLD"],
                    )
                )
            except Exception as e:
                logger.error(
                    f"Inference error detected: {e}. Reinitializing detector..."
                )
                with self.detector_lock:
                    try:
                        self.detector_instance = Detector(
                            model_choice=self.model_choice, debug=self.debug
                        )
                        model_id = getattr(self.detector_instance, "model_id", "") or ""
                        if not model_id and hasattr(self.detector_instance, "model_path"):
                            model_id = os.path.basename(self.detector_instance.model_path)
                        self.detector_model_id = model_id
                    except Exception as e2:
                        logger.error(f"Detector reinitialization failed: {e2}")
                time.sleep(1)
                continue

            with self.frame_lock:
                self.latest_detection_time = time.time()

            if object_detected:
                self.last_detection_event_time = time.time()
                self.daylight_no_detection_seconds = 0.0
                self.last_daylight_check_ts = self.last_detection_event_time
                if self.no_detection_alert_active:
                    self.no_detection_alert_active = False
                self._enqueue_processing_job(
                    {
                        "capture_time_precise": capture_time_precise,
                        "original_frame": original_frame,
                        "detection_info_list": detection_info_list,
                    }
                )

            self._maybe_send_status_alerts()

            detection_time = time.time() - start_time
            target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
            sleep_time = target_duration - detection_time
            # Ensures a minimal sleep time to reduce CPU usage even if detection takes longer than target
            if sleep_time <= 0:
                sleep_time = 0.01  # Minimal sleep duration in seconds
            logger.info(
                f"AI duration: {detection_time:.4f}s, sleeping for: {sleep_time:.4f}s"
            )
            time.sleep(sleep_time)

        logger.info("Detection loop stopped.")

    def _enqueue_processing_job(self, job):
        """Enqueues a processing job, dropping the oldest if the queue is full."""
        try:
            self.processing_queue.put_nowait(job)
        except queue.Full:
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.processing_queue.put_nowait(job)
            except queue.Full:
                logger.warning("Processing queue full; dropped latest job.")

    def _processing_loop(self):
        """Handles I/O heavy work off the detection loop."""
        logger.info("Processing loop (worker) started.")
        while not self.stop_event.is_set():
            try:
                job = self.processing_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            capture_time_precise = job.get("capture_time_precise")
            original_frame = job.get("original_frame")
            detection_info_list = job.get("detection_info_list", [])

            if original_frame is None or not detection_info_list:
                continue

            timestamp_str = capture_time_precise.strftime("%Y%m%d_%H%M%S")
            day_folder = os.path.join(self.output_dir, timestamp_str[:8])
            os.makedirs(day_folder, exist_ok=True)
            csv_path = os.path.join(day_folder, "images.csv")

            best_det = max(detection_info_list, key=lambda d: d["confidence"])
            best_class = best_det["class_name"]
            best_class_sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", best_class)
            best_class_conf = best_det["confidence"]

            original_name = f"{timestamp_str}_{best_class_sanitized}_original.jpg"
            optimized_name = f"{timestamp_str}_{best_class_sanitized}_optimized.jpg"
            zoomed_name = f"{timestamp_str}_{best_class_sanitized}_zoomed.jpg"

            original_path = os.path.join(day_folder, original_name)
            optimized_path = os.path.join(day_folder, optimized_name)
            zoomed_path = os.path.join(day_folder, zoomed_name)

            image_id = int(timestamp_str.replace("_", ""))
            image_info = {
                "id": image_id,
                "file_name": original_name,
                "width": original_frame.shape[1],
                "height": original_frame.shape[0],
            }
            annotations = []
            for i, det in enumerate(detection_info_list, start=1):
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                area = (x2 - x1) * (y2 - y1)
                category_id = (
                    7
                    if det["class_name"].replace(" ", "_") == best_class_sanitized
                    else 1
                )
                annotations.append(
                    {
                        "id": image_id * 100 + i,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }
                )
            categories = []
            unique_categories = {}
            for det in detection_info_list:
                cat_name = det["class_name"].replace(" ", "_")
                if cat_name not in unique_categories:
                    cat_id = 7 if cat_name == best_class_sanitized else 1
                    unique_categories[cat_name] = cat_id
                    categories.append({"id": cat_id, "name": cat_name})
            coco_detection = {
                "annotations": annotations,
                "images": [image_info],
                "categories": categories,
            }
            coco_json = json.dumps(coco_detection)

            save_success_original = False
            save_success_optimized = False
            actual_zoomed_path = None

            top1_class_name = ""
            top1_confidence = ""
            zoomed_frame = None
            if detection_info_list:
                try:
                    bbox = (
                        best_det["x1"],
                        best_det["y1"],
                        best_det["x2"],
                        best_det["y2"],
                    )
                    zoomed_frame_raw = self.create_square_crop(
                        original_frame, bbox, margin_percent=0.1
                    )

                    if zoomed_frame_raw is not None and zoomed_frame_raw.size > 0:
                        zoomed_frame = cv2.resize(
                            zoomed_frame_raw,
                            (
                                self.SAVE_RESOLUTION_ZOOMED,
                                self.SAVE_RESOLUTION_ZOOMED,
                            ),
                        )

                        try:
                            zoomed_frame_rgb = cv2.cvtColor(
                                zoomed_frame, cv2.COLOR_BGR2RGB
                            )
                            _, _, top1_class_name, top1_confidence = (
                                self.classifier.predict_from_image(zoomed_frame_rgb)
                            )
                        except Exception as e:
                            logger.error(f"Classification failed: {e}")
                            top1_class_name = "CLASSIFICATION_ERROR"
                            top1_confidence = 0.0

                    else:
                        logger.warning(
                            "Zoomed frame generation failed or resulted in empty image. Skipping save."
                        )
                        zoomed_name = ""
                except cv2.error as e:
                    logger.error(
                        f"OpenCV error during zoomed image processing/saving: {e}"
                    )
                    zoomed_name = ""
                except Exception as e:
                    logger.error(
                        f"Unexpected error during zoomed image processing/saving: {e}"
                    )
                    zoomed_name = ""

            logger.debug(f"Classification Result: {top1_class_name} - {top1_confidence}")

            should_save = True
            if self.config.get("SAVE_REQUIRES_CLASSIFIER", False):
                try:
                    classifier_confidence = float(top1_confidence)
                except Exception:
                    classifier_confidence = 0.0
                detector_label = str(best_class_sanitized).strip().lower()
                classifier_label = (
                    str(top1_class_name or "").replace(" ", "_").strip().lower()
                )
                classifier_threshold = float(
                    self.config.get("CLASSIFIER_CONFIDENCE_THRESHOLD", 0.55)
                )
                should_save = (
                    classifier_confidence >= classifier_threshold
                    and classifier_label == detector_label
                )
                if not should_save:
                    logger.info(
                        "Classifier gate skipped save (detector=%s, classifier=%s %.2f < %.2f).",
                        detector_label,
                        classifier_label,
                        classifier_confidence,
                        classifier_threshold,
                    )
            if not should_save:
                continue

            try:
                save_success_original = cv2.imwrite(original_path, original_frame)
                if save_success_original:
                    add_exif_metadata(
                        original_path, capture_time_precise, self.location_config
                    )
                else:
                    logger.error(
                        f"Failed to save original image (imwrite returned false): {original_path}"
                    )
            except Exception as e:
                logger.error(f"Error during original image saving/EXIF: {e}")
                save_success_original = False

            try:
                if original_frame.shape[1] > 800:
                    optimized_frame = cv2.resize(
                        original_frame,
                        (
                            800,
                            int(original_frame.shape[0] * 800 / original_frame.shape[1]),
                        ),
                    )
                    save_success_optimized = cv2.imwrite(
                        optimized_path,
                        optimized_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                    )
                else:
                    save_success_optimized = cv2.imwrite(
                        optimized_path,
                        original_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                    )

                if save_success_optimized:
                    add_exif_metadata(
                        optimized_path, capture_time_precise, self.location_config
                    )
                else:
                    logger.error(f"Failed to save optimized image: {optimized_path}")
            except cv2.error as e:
                logger.error(f"OpenCV error during optimized image processing/saving: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error during optimized image processing/saving: {e}"
                )

            if zoomed_frame is not None:
                save_success_zoomed = cv2.imwrite(
                    zoomed_path,
                    zoomed_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                )
                if save_success_zoomed:
                    add_exif_metadata(
                        zoomed_path, capture_time_precise, self.location_config
                    )
                    actual_zoomed_path = zoomed_path
                else:
                    logger.error(f"Failed to save zoomed image: {zoomed_path}")
                    zoomed_name = ""

            try:
                insert_image(
                    self.db_conn,
                    {
                        "timestamp": timestamp_str,
                        "original_name": original_name,
                        "optimized_name": optimized_name,
                        "zoomed_name": zoomed_name,
                        "best_class": best_class_sanitized,
                        "best_class_conf": best_class_conf,
                        "top1_class_name": top1_class_name,
                        "top1_confidence": top1_confidence,
                        "coco_json": coco_json,
                        "downloaded_timestamp": "",
                        "detector_model_id": self.detector_model_id,
                        "classifier_model_id": self.classifier_model_id,
                    },
                )
            except Exception as e:
                logger.error(f"Error writing detection to SQLite: {e}")

            if self.telegram_rule != "basic":
                continue

            # Telegram notification (best-effort).
            self.detection_occurred = True
            self.detection_counter += len(detection_info_list)
            self.detection_classes_agg.update(
                det["class_name"] for det in detection_info_list
            )

            current_time = time.time()
            cooldown = self.config.get("TELEGRAM_COOLDOWN", 60)

            if self.detection_occurred and (
                current_time - self.last_notification_time >= cooldown
            ):
                if not self.config.get("TELEGRAM_ENABLED", True):
                    self.last_notification_time = current_time
                    self.detection_occurred = False
                    self.detection_counter = 0
                    self.detection_classes_agg = set()
                    continue
                with self.telegram_lock:
                    if self.detection_occurred and (
                        current_time - self.last_notification_time >= cooldown
                    ):
                        aggregated_classes = ", ".join(
                            sorted(self.detection_classes_agg)
                        )
                        alert_text = (
                            "Пуп!"
                            f"Ага: {aggregated_classes}"
                            f"Тут ({len(detection_info_list)} текст {self.detection_counter} и вот)"
                        )

                        photo_to_send = None
                        if actual_zoomed_path and os.path.exists(actual_zoomed_path):
                            photo_to_send = actual_zoomed_path
                        elif save_success_optimized and os.path.exists(
                            optimized_path
                        ):
                            photo_to_send = optimized_path
                        elif save_success_original and os.path.exists(original_path):
                            photo_to_send = original_path

                        if photo_to_send:
                            try:
                                send_telegram_message(
                                    text=alert_text, photo_path=photo_to_send
                                )
                                logger.info(
                                    f"(Simulated) Telegram notification sent: {alert_text} with {os.path.basename(photo_to_send)}"
                                )
                                self.last_notification_time = current_time
                                self.detection_occurred = False
                                self.detection_counter = 0
                                self.detection_classes_agg = set()
                            except Exception as e:
                                logger.error(f"Failed to send Telegram notification: {e}")
                        else:
                            logger.warning(
                                "No suitable image found to send with Telegram alert. Sending text only."
                            )
                            try:
                                send_telegram_message(text=alert_text)
                                logger.info(
                                    f"(Simulated) Telegram notification sent (text only): {alert_text}"
                                )
                                self.last_notification_time = current_time
                                self.detection_occurred = False
                                self.detection_counter = 0
                                self.detection_classes_agg = set()
                            except Exception as e:
                                logger.error(
                                    f"Failed to send Telegram text-only notification: {e}"
                                )

        logger.info("Processing loop stopped.")

    # >>> Get Display Frame Function >>>
    def get_display_frame(self):
        """
        Returns the most recent frame to be displayed.

        Args:
            None

        Returns:
            np.ndarray or None: The latest frame or None if not available.
        """
        with self.frame_lock:
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            else:
                return None

    # >>> Start DetectionManager >>>
    def start(self):
        """
        Starts the DetectionManager by initializing components and starting threads.

        Args:
            None

        Returns:
            None
        """
        self.frame_thread.start()
        self.detection_thread.start()
        self.processing_thread.start()
        logger.info("DetectionManager started.")

    # >>> Stop DetectionManager >>>
    def stop(self):
        """
        Stops the DetectionManager and releases resources.

        Args:
            None

        Returns:
            None
        """
        self.stop_event.set()
        if hasattr(self, "frame_thread") and self.frame_thread.is_alive():
            self.frame_thread.join()
        if hasattr(self, "detection_thread") and self.detection_thread.is_alive():
            self.detection_thread.join()
        if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
            self.processing_thread.join()
        if self.video_capture:
            self.video_capture.stop()
        try:
            if self.db_conn:
                self.db_conn.close()
        except Exception:
            pass
        logger.info("DetectionManager stopped and video capture released.")
