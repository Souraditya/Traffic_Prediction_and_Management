"""
collectors/weather_events_collector.py
---------------------------------------
Step 4 of Data Acquisition: External Contextual Data

Weather has a large impact on traffic:
  - Rain  : -15 to -30% average speed
  - Fog   : visibility < 200m → -40% speed
  - Flood : road closures → complete rerouting

Events (concerts, sports, rallies) create predictable
demand spikes that a time-series model alone would miss.

Data sources
-------------
  Weather : OpenMeteo API (completely free, no API key)
  Events  : PredictHQ (commercial) or manual calendar
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from loguru import logger

from config.settings import settings
from utils.helpers import utc_now, append_csv, safe_get, publish_to_kafka


# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class WeatherReading:
    timestamp:         str
    latitude:          float
    longitude:         float
    zone_id:           str
    temperature_c:     float
    precipitation_mm:  float       # mm in the last hour
    visibility_km:     float
    wind_speed_kmh:    float
    weather_code:      int         # WMO code (0=clear, 61=rain, 71=snow, 95=storm)
    weather_desc:      str
    # Derived impact score: 0.0 (no impact) → 1.0 (severe impact on traffic)
    traffic_impact:    float = 0.0

    # WMO weather code descriptions (subset)
    WMO_CODES: Dict = None

    def __post_init__(self):
        self.WMO_CODES = {
            0: "clear", 1: "mainly_clear", 2: "partly_cloudy", 3: "overcast",
            45: "fog", 48: "icy_fog",
            51: "light_drizzle", 53: "drizzle", 55: "heavy_drizzle",
            61: "light_rain", 63: "rain", 65: "heavy_rain",
            71: "light_snow", 73: "snow", 75: "heavy_snow",
            80: "rain_showers", 81: "heavy_showers",
            95: "thunderstorm", 96: "hail_storm",
        }
        self.weather_desc   = self.WMO_CODES.get(self.weather_code, "unknown")
        self.traffic_impact = self._compute_impact()

    def _compute_impact(self) -> float:
        """
        Estimate traffic impact (0–1) from weather conditions.
        Based on empirical traffic engineering studies.
        """
        impact = 0.0

        # Rain impact: 0.1 (drizzle) → 0.3 (heavy rain)
        if self.precipitation_mm > 10:
            impact += 0.30
        elif self.precipitation_mm > 5:
            impact += 0.20
        elif self.precipitation_mm > 1:
            impact += 0.10
        elif self.precipitation_mm > 0:
            impact += 0.05

        # Visibility impact: severe below 200m, moderate below 1km
        if self.visibility_km < 0.2:
            impact += 0.40
        elif self.visibility_km < 1.0:
            impact += 0.20
        elif self.visibility_km < 4.0:
            impact += 0.05

        # Storm / thunderstorm
        if self.weather_code >= 95:
            impact += 0.25

        # Snow
        if 70 <= self.weather_code <= 79:
            impact += 0.35

        return round(min(1.0, impact), 3)


@dataclass
class TrafficEvent:
    event_id:    str
    timestamp:   str
    event_type:  str          # concert, sports, rally, parade, incident
    title:       str
    venue_name:  str
    latitude:    float
    longitude:   float
    start_time:  str
    end_time:    str
    expected_attendance: int
    radius_impact_km: float   # affected radius around venue
    severity:    str = "medium"   # low / medium / high


# ── Weather collector ─────────────────────────────────────────────────────────
class WeatherCollector:
    """
    Fetches weather data from OpenMeteo (free, no API key required).
    Covers current conditions + 24h forecast.
    """

    def __init__(self):
        self.cfg     = settings.weather
        self._output = self.cfg.output_dir / "weather.csv"
        self._output.parent.mkdir(parents=True, exist_ok=True)
        # Divide city into zones (use more zones for large cities)
        self.zones = {
            "north": (settings.latitude + 0.05, settings.longitude),
            "south": (settings.latitude - 0.05, settings.longitude),
            "east":  (settings.latitude,         settings.longitude + 0.05),
            "west":  (settings.latitude,         settings.longitude - 0.05),
            "centre":(settings.latitude,         settings.longitude),
        }

    def fetch_zone(self, zone_id: str, lat: float, lon: float) -> Optional[WeatherReading]:
        """Fetch current weather for a single zone."""
        params = {
            "latitude":  lat,
            "longitude": lon,
            "current":   ",".join(self.cfg.variables),
            "timezone":  "Asia/Kolkata",
        }

        data = safe_get(self.cfg.base_url, params=params)
        if not data:
            return None

        try:
            current = data.get("current", {})
            reading = WeatherReading(
                timestamp        = current.get("time", utc_now()),
                latitude         = lat,
                longitude        = lon,
                zone_id          = zone_id,
                temperature_c    = float(current.get("temperature_2m",   20)),
                precipitation_mm = float(current.get("precipitation",     0)),
                visibility_km    = float(current.get("visibility",       10)) / 1000,
                wind_speed_kmh   = float(current.get("wind_speed_10m",    0)),
                weather_code     = int(current.get("weather_code",        0)),
                weather_desc     = "",          # filled in __post_init__
                traffic_impact   = 0.0,         # filled in __post_init__
            )
            return reading
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Weather parse error for zone {zone_id}: {e}")
            return None

    def fetch_all_zones(self) -> List[WeatherReading]:
        """Fetch weather for all city zones."""
        readings = []
        for zone_id, (lat, lon) in self.zones.items():
            reading = self.fetch_zone(zone_id, lat, lon)
            if reading:
                readings.append(reading)
                self._save(reading)
                logger.debug(f"Weather {zone_id}: {reading.weather_desc}, "
                             f"rain={reading.precipitation_mm}mm, "
                             f"impact={reading.traffic_impact:.0%}")
        return readings

    def fetch_forecast(self, hours: int = 24) -> List[Dict]:
        """
        Fetch hourly forecast for the next N hours.
        Used by the ML model to anticipate weather-related congestion.
        """
        params = {
            "latitude":  settings.latitude,
            "longitude": settings.longitude,
            "hourly":    ",".join(self.cfg.variables),
            "forecast_days": max(1, hours // 24 + 1),
            "timezone":  "Asia/Kolkata",
        }
        data = safe_get(self.cfg.base_url, params=params)
        if not data:
            return []

        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])[:hours]
        result = []

        for i, ts in enumerate(times):
            result.append({
                "timestamp":       ts,
                "temperature_c":   hourly.get("temperature_2m",  [None])[i],
                "precipitation_mm":hourly.get("precipitation",   [0])[i],
                "visibility_m":    hourly.get("visibility",      [10000])[i],
                "wind_speed_kmh":  hourly.get("wind_speed_10m",  [0])[i],
                "weather_code":    hourly.get("weather_code",    [0])[i],
            })

        logger.info(f"Fetched {len(result)}-hour weather forecast")
        return result

    def run_polling_loop(self, stop_event: threading.Event = None) -> None:
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            self.fetch_all_zones()
            stop_event.wait(self.cfg.poll_interval_sec)

    def _save(self, reading: WeatherReading) -> None:
        row = {k: v for k, v in asdict(reading).items() if k != "WMO_CODES"}
        append_csv(row, self._output)

        if settings.kafka_enabled:
            publish_to_kafka(
                topic             = settings.kafka.topics["weather"],
                payload           = row,
                bootstrap_servers = settings.kafka.bootstrap_servers,
            )


# ── Events collector ──────────────────────────────────────────────────────────
class EventsCollector:
    """
    Collects planned events that will affect traffic demand.

    Integration options
    -------------------
    1. PredictHQ API     – paid, very accurate event data for India
    2. Ticketmaster API  – concerts + sports (free tier)
    3. Manual CSV        – municipal events calendar (parades, elections)
    4. Synthetic         – generates realistic test events
    """

    def __init__(self):
        self._output = settings.gps.output_dir.parent / "events" / "events.csv"
        self._output.parent.mkdir(parents=True, exist_ok=True)

    def load_manual_calendar(self, csv_path: Path) -> List[TrafficEvent]:
        """
        Load planned events from a manually maintained CSV.
        Expected columns: event_id, event_type, title, venue_name,
                          latitude, longitude, start_time, end_time,
                          expected_attendance
        """
        import csv as csv_mod
        events = []
        try:
            with open(csv_path, "r") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    evt = TrafficEvent(
                        event_id    = row["event_id"],
                        timestamp   = utc_now(),
                        event_type  = row.get("event_type", "event"),
                        title       = row["title"],
                        venue_name  = row.get("venue_name", ""),
                        latitude    = float(row["latitude"]),
                        longitude   = float(row["longitude"]),
                        start_time  = row["start_time"],
                        end_time    = row["end_time"],
                        expected_attendance = int(row.get("expected_attendance", 1000)),
                        radius_impact_km    = float(row.get("radius_km", 1.5)),
                        severity    = self._attendance_to_severity(
                                        int(row.get("expected_attendance", 1000))),
                    )
                    events.append(evt)
                    self._save_event(evt)
            logger.info(f"Loaded {len(events)} events from {csv_path}")
        except FileNotFoundError:
            logger.warning(f"Events calendar not found: {csv_path}")
        return events

    def generate_synthetic_events(self) -> List[TrafficEvent]:
        """
        Synthetic events for Kolkata – typical weekly programme.
        Useful for testing event-impact features in the ML model.
        """
        from datetime import datetime, timedelta
        import random

        today = datetime.now()
        events_data = [
            {
                "type": "sports", "title": "Eden Gardens Cricket",
                "venue": "Eden Gardens",
                "lat": 22.5645, "lon": 88.3432,
                "attendance": 65000, "dur_hr": 6
            },
            {
                "type": "concert", "title": "Rabindra Sadan Concert",
                "venue": "Rabindra Sadan",
                "lat": 22.5448, "lon": 88.3426,
                "attendance": 2000, "dur_hr": 3
            },
            {
                "type": "rally", "title": "Brigade Parade Ground Rally",
                "venue": "Brigade Parade Ground",
                "lat": 22.5592, "lon": 88.3494,
                "attendance": 100000, "dur_hr": 4
            },
            {
                "type": "festival", "title": "Science City Exhibition",
                "venue": "Science City",
                "lat": 22.5339, "lon": 88.3962,
                "attendance": 5000, "dur_hr": 8
            },
        ]

        events = []
        for i, e in enumerate(events_data):
            start = today.replace(hour=18, minute=0, second=0) + \
                    timedelta(days=random.randint(0, 6))
            end   = start + timedelta(hours=e["dur_hr"])

            evt = TrafficEvent(
                event_id    = f"evt_{i+1:04d}",
                timestamp   = utc_now(),
                event_type  = e["type"],
                title       = e["title"],
                venue_name  = e["venue"],
                latitude    = e["lat"],
                longitude   = e["lon"],
                start_time  = start.isoformat(),
                end_time    = end.isoformat(),
                expected_attendance = e["attendance"],
                radius_impact_km    = self._attendance_to_radius(e["attendance"]),
                severity    = self._attendance_to_severity(e["attendance"]),
            )
            events.append(evt)
            self._save_event(evt)

        logger.info(f"Generated {len(events)} synthetic events")
        return events

    @staticmethod
    def _attendance_to_severity(attendance: int) -> str:
        if attendance >= 50000: return "high"
        if attendance >= 5000:  return "medium"
        return "low"

    @staticmethod
    def _attendance_to_radius(attendance: int) -> float:
        """Approximate traffic impact radius (km) based on attendance."""
        if attendance >= 100000: return 5.0
        if attendance >= 50000:  return 3.0
        if attendance >= 10000:  return 2.0
        return 1.0

    def _save_event(self, evt: TrafficEvent) -> None:
        append_csv(asdict(evt), self._output)

        if settings.kafka_enabled:
            publish_to_kafka(
                topic             = settings.kafka.topics["events"],
                payload           = asdict(evt),
                bootstrap_servers = settings.kafka.bootstrap_servers,
            )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Weather Collector ===")
    wc = WeatherCollector()
    readings = wc.fetch_all_zones()

    if readings:
        for r in readings:
            print(f"  Zone {r.zone_id:8s}: {r.weather_desc:<15} "
                  f"rain={r.precipitation_mm:.1f}mm "
                  f"impact={r.traffic_impact:.0%}")
    else:
        print("  (OpenMeteo unreachable – check network)")

    print("\n=== Events Collector ===")
    ec = EventsCollector()
    events = ec.generate_synthetic_events()
    for e in events:
        print(f"  [{e.severity.upper():6s}] {e.title:<35} "
              f"r={e.radius_impact_km}km  att={e.expected_attendance:,}")

    print(f"\nWeather → {wc._output}")
    print(f"Events  → {ec._output}")