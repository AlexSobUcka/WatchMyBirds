from datetime import datetime

import pytz
from astral import Observer
from astral.geocoder import database, lookup
from astral.sun import sun

from config import get_config


def main():
    """Zeigt die aktuelle Day/Night-Berechnung fuer die Detektion."""
    config = get_config()
    location = str(config.get("DAY_AND_NIGHT_CAPTURE_LOCATION", "Berlin")).strip()
    if not location:
        location = "Berlin"
    tz_value = str(config.get("TELEGRAM_TIMEZONE", "")).strip()
    tz_override = None
    if tz_value:
        try:
            tz_override = pytz.timezone(tz_value)
        except Exception as exc:
            print(f"Invalid TELEGRAM_TIMEZONE '{tz_value}': {exc}")

    try:
        if "," in location:
            lat_str, lon_str = location.split(",", 1)
            observer = Observer(latitude=float(lat_str), longitude=float(lon_str))
            tz = tz_override or pytz.UTC
            display_location = (
                f"{location} ({tz.zone})" if tz_override else f"{location} (UTC)"
            )
        else:
            city = lookup(location, database())
            tz = pytz.timezone(city.timezone)
            observer = city.observer
            display_location = f"{location} ({city.timezone})"
        now = datetime.now(tz)
        s = sun(observer, date=now, tzinfo=tz)
    except Exception as exc:
        print(f"Failed to resolve daylight times for '{location}': {exc}")
        return

    is_day = s["sunrise"] < now < s["sunset"]
    gate_enabled = bool(config.get("DAY_AND_NIGHT_CAPTURE", True))

    print(f"Location: {display_location}")
    print(f"Now:      {now}")
    print(f"Sunrise:  {s['sunrise']}")
    print(f"Sunset:   {s['sunset']}")
    print(f"Is day:   {is_day}")
    print(f"Gate enabled: {gate_enabled}")
    if gate_enabled:
        print("Detection runs only when Is day is True.")
    else:
        print("Detection runs regardless of Is day (gate bypass).")


if __name__ == "__main__":
    main()
