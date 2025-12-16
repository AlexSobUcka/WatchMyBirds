# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file once.
load_dotenv()


_CONFIG = None


def _load_config():
    """Lädt Konfiguration aus Umgebungsvariablen."""
    location_str = os.getenv("LOCATION_DATA", "52.516, 13.377")
    try:
        lat_str, lon_str = location_str.split(",")
        location_data = {"latitude": float(lat_str), "longitude": float(lon_str)}
    except Exception:
        location_data = {"latitude": 52.516, "longitude": 13.377}

    config = {
        # General Settings
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "False").lower() == "true",
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "/output"),
        "VIDEO_SOURCE": os.getenv("VIDEO_SOURCE", "0"),
        # GPS Location
        "LOCATION_DATA": location_data,
        # Model and Detection Settings
        "DETECTOR_MODEL_CHOICE": os.getenv("DETECTOR_MODEL_CHOICE", "yolo"),
        "CONFIDENCE_THRESHOLD_DETECTION": float(
            os.getenv("CONFIDENCE_THRESHOLD_DETECTION", 0.55)
        ),
        "SAVE_THRESHOLD": float(os.getenv("SAVE_THRESHOLD", 0.55)),
        "MAX_FPS_DETECTION": float(os.getenv("MAX_FPS_DETECTION", 0.5)),
        "MODEL_BASE_PATH": os.getenv("MODEL_BASE_PATH", "/models"),
        # Model and Classifier Settings
        "CLASSIFIER_CONFIDENCE_THRESHOLD": float(
            os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", 0.55)
        ),
        # Results Settings
        "FUSION_ALPHA": float(os.getenv("FUSION_ALPHA", 0.5)),
        # Streaming Settings
        "STREAM_FPS": float(os.getenv("STREAM_FPS", 0)),  # UI throttling
        "STREAM_FPS_CAPTURE": float(os.getenv("STREAM_FPS_CAPTURE", 0)),  # reader throttling
        "STREAM_WIDTH_OUTPUT_RESIZE": int(os.getenv("STREAM_WIDTH_OUTPUT_RESIZE", 640)),
        # Day and Night Capture Settings
        "DAY_AND_NIGHT_CAPTURE": os.getenv("DAY_AND_NIGHT_CAPTURE", "True").lower()
        == "true",
        "DAY_AND_NIGHT_CAPTURE_LOCATION": os.getenv(
            "DAY_AND_NIGHT_CAPTURE_LOCATION", "Berlin"
        ),
        # CPU and Resource Management
        "CPU_LIMIT": int(float(os.getenv("CPU_LIMIT", 1))),
        # Telegram Notification Settings
        "TELEGRAM_COOLDOWN": float(os.getenv("TELEGRAM_COOLDOWN", 5)),
        "EDIT_PASSWORD": os.getenv("EDIT_PASSWORD", "SECRET_PASSWORD"),
    }
    return config


def get_config():
    """Gibt die einmal geladene Konfiguration zurück."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
        _coerce_config_types(_CONFIG)
    return _CONFIG


def _coerce_config_types(config):
    """Validiert und erzwingt erwartete Typen für zentrale Keys."""
    # VIDEO_SOURCE: int for webcams, string otherwise (startup-only per locked decision).
    source = config.get("VIDEO_SOURCE", "0")
    try:
        if str(source).isdigit():
            config["VIDEO_SOURCE"] = int(source)
    except Exception:
        config["VIDEO_SOURCE"] = source

    # STREAM_FPS / STREAM_FPS_CAPTURE: allow 0 to disable throttling; otherwise positive float
    try:
        stream_fps = float(config.get("STREAM_FPS", 1))
        config["STREAM_FPS"] = stream_fps if stream_fps > 0 else 0.0
    except Exception:
        config["STREAM_FPS"] = 0.0
    try:
        stream_fps_capture = float(config.get("STREAM_FPS_CAPTURE", 0))
        config["STREAM_FPS_CAPTURE"] = (
            stream_fps_capture if stream_fps_capture > 0 else 0.0
        )
    except Exception:
        config["STREAM_FPS_CAPTURE"] = 0.0

    # CPU_LIMIT should be positive int; fallback to 1
    try:
        cpu_limit = int(float(config.get("CPU_LIMIT", 1)))
        config["CPU_LIMIT"] = cpu_limit if cpu_limit > 0 else 1
    except Exception:
        config["CPU_LIMIT"] = 1

    # Thresholds: guard against invalid values
    for key in ("CONFIDENCE_THRESHOLD_DETECTION", "SAVE_THRESHOLD"):
        try:
            val = float(config.get(key, 0.55))
            config[key] = max(0.0, min(1.0, val))
        except Exception:
            config[key] = 0.55


# Backward-compatible alias
def load_config():
    """Alias für Alt-Code; liefert die geteilte Konfiguration."""
    return get_config()


if __name__ == "__main__":
    from pprint import pprint

    pprint(get_config())
