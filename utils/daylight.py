import time
from datetime import datetime

import pytz
from astral import Observer
from astral.geocoder import database, lookup
from astral.sun import sun

from logging_config import get_logger

logger = get_logger(__name__)


def _parse_location_to_observer(location_value):
    location_text = str(location_value or "").strip()
    if not location_text:
        return None
    if "," in location_text:
        try:
            lat_str, lon_str = location_text.split(",", 1)
            latitude = float(lat_str)
            longitude = float(lon_str)
            return Observer(latitude=latitude, longitude=longitude)
        except Exception:
            return None
    try:
        city = lookup(location_text, database())
        return city.observer
    except Exception:
        return None


def is_daytime(location_value, cache, ttl_seconds=300, tz_value=None):
    """Prueft Tageslicht anhand Sunrise/Sunset mit Cache."""
    try:
        now_ts = time.time()
        if (
            cache.get("city") == location_value
            and (now_ts - cache.get("ts", 0.0)) < ttl_seconds
        ):
            return cache.get("value", True)

        observer = _parse_location_to_observer(location_value)
        if observer is None:
            raise ValueError("Invalid location for daylight calculation.")
        tz = None
        if isinstance(tz_value, str) and tz_value.strip():
            try:
                tz = pytz.timezone(tz_value.strip())
            except Exception:
                tz = None
        elif tz_value is not None:
            tz = tz_value

        if isinstance(location_value, str) and "," not in location_value:
            city = lookup(location_value.strip(), database())
            tz = pytz.timezone(city.timezone)
        if tz is None:
            tz = pytz.UTC
        now = datetime.now(tz)
        s = sun(observer, date=now, tzinfo=tz)
        value = s["sunrise"] < now < s["sunset"]
        cache.update({"city": location_value, "value": value, "ts": now_ts})
        return value
    except Exception as exc:
        logger.error(f"Error determining daylight status: {exc}")
        cache.update({"city": location_value, "value": True, "ts": time.time()})
        return True
