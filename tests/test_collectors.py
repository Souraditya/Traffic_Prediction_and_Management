"""
tests/test_collectors.py
-------------------------
Unit tests for all data acquisition collectors.
All tests run WITHOUT external APIs (pure unit tests).

Run:  pytest tests/ -v
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


# ── Camera collector tests ────────────────────────────────────────────────────
class TestCameraCollector:

    def setup_method(self):
        from collectors.camera_collector import CameraCollector
        self.collector = CameraCollector()

    def test_simulate_day_returns_dataframe(self):
        df = self.collector.simulate_day(camera_id="cam_001")
        assert len(df) > 0
        assert "cars" in df.columns
        assert "total" in df.columns

    def test_simulate_day_rush_hour_higher_counts(self):
        """Rush-hour records should have higher vehicle counts than night records."""
        df = self.collector.simulate_day()
        rush  = df[df["timestamp"].str[11:13].astype(int).between(7, 9)]["total"].mean()
        night = df[df["timestamp"].str[11:13].astype(int).between(1, 4)]["total"].mean()
        assert rush > night, f"Rush ({rush:.1f}) should exceed night ({night:.1f})"

    def test_edge_payload_parsing(self):
        payload = {
            "camera_id": "cam_001",
            "timestamp": "2026-03-26T08:30:00Z",
            "counts": {"car": 12, "motorcycle": 5, "bus": 1, "truck": 0},
            "incident": False,
            "confidence": 0.88,
        }
        record = self.collector.process_edge_payload(payload)
        assert record is not None
        assert record.cars == 12
        assert record.motorcycles == 5
        assert record.total == 18
        assert record.incident_flag is False

    def test_edge_payload_missing_key(self):
        record = self.collector.process_edge_payload({"foo": "bar"})
        # Should handle gracefully (no crash, returns None or partial)
        # camera_id will be empty, but no KeyError crash
        assert True  # test passes if no exception raised

    def test_output_csv_created_after_simulate(self):
        self.collector.simulate_day()
        assert self.collector._output_csv.exists()

    def test_vehicle_total_equals_sum(self):
        from collectors.camera_collector import VehicleCounts
        vc = VehicleCounts(
            camera_id="test", timestamp="2026-01-01T00:00:00Z",
            latitude=0, longitude=0,
            cars=10, motorcycles=5, buses=2, trucks=1
        )
        assert vc.total == 18


# ── Sensor collector tests ────────────────────────────────────────────────────
class TestSensorCollector:

    def setup_method(self):
        from collectors.sensor_collector import SensorCollector, RESTSensorPoller
        self.collector = SensorCollector()
        self.poller    = RESTSensorPoller()   # no API URL = synthetic mode

    def test_snapshot_returns_readings(self):
        readings = self.collector.collect_snapshot()
        assert len(readings) > 0

    def test_synthetic_reading_ranges(self):
        from collectors.sensor_collector import SENSOR_REGISTRY
        reading = self.poller._generate_synthetic("loop_001")
        assert 0 <= reading.speed_kmh <= 130
        assert 0 <= reading.occupancy_pct <= 1.0
        assert reading.flow_veh_per_hr >= 0

    def test_congestion_level_mapping(self):
        from collectors.sensor_collector import SensorReading
        r = SensorReading("s", "ts", 0, 0, "road", 5, 0.9, 50)
        assert r.congestion_level == "gridlock"

        r.speed_kmh = 20
        assert r.congestion_level == "heavy"

        r.speed_kmh = 40
        assert r.congestion_level == "moderate"

        r.speed_kmh = 55
        assert r.congestion_level == "light"

        r.speed_kmh = 80
        assert r.congestion_level == "free_flow"

    def test_mqtt_payload_parsing(self):
        from collectors.sensor_collector import MQTTSensorSubscriber

        received = []
        sub = MQTTSensorSubscriber(on_reading=received.append)

        payload = {
            "sensor_id": "loop_001",
            "ts": int(datetime.now(timezone.utc).timestamp() * 1000),
            "speed": 45.2,
            "occupancy": 0.55,
            "flow": 900,
            "headway": 4.0,
        }
        result = sub._parse_payload(payload)
        assert result is not None
        assert result.speed_kmh == 45.2
        assert result.quality_flag == 1

    def test_out_of_range_flagged(self):
        from collectors.sensor_collector import MQTTSensorSubscriber
        sub = MQTTSensorSubscriber(on_reading=lambda x: None)
        payload = {
            "sensor_id": "loop_001",
            "ts": 1711436400000,
            "speed": 999,       # clearly wrong
            "occupancy": 5.0,   # out of range
            "flow": 99999,
        }
        result = sub._parse_payload(payload)
        assert result.quality_flag == 0


# ── Weather collector tests ───────────────────────────────────────────────────
class TestWeatherCollector:

    def setup_method(self):
        from collectors.weather_events_collector import WeatherCollector
        self.collector = WeatherCollector()

    def test_weather_reading_impact_clear(self):
        from collectors.weather_events_collector import WeatherReading
        r = WeatherReading(
            timestamp="ts", latitude=22.5, longitude=88.3, zone_id="centre",
            temperature_c=28, precipitation_mm=0, visibility_km=10,
            wind_speed_kmh=10, weather_code=0, weather_desc="", traffic_impact=0.0
        )
        assert r.traffic_impact == 0.0
        assert r.weather_desc == "clear"

    def test_weather_reading_impact_heavy_rain(self):
        from collectors.weather_events_collector import WeatherReading
        r = WeatherReading(
            timestamp="ts", latitude=22.5, longitude=88.3, zone_id="centre",
            temperature_c=22, precipitation_mm=15, visibility_km=2,
            wind_speed_kmh=30, weather_code=65, weather_desc="", traffic_impact=0.0
        )
        assert r.traffic_impact > 0.3  # heavy rain should have significant impact
        assert r.weather_desc == "heavy_rain"

    def test_weather_reading_fog_impact(self):
        from collectors.weather_events_collector import WeatherReading
        r = WeatherReading(
            timestamp="ts", latitude=22.5, longitude=88.3, zone_id="centre",
            temperature_c=18, precipitation_mm=0, visibility_km=0.1,
            wind_speed_kmh=5, weather_code=45, weather_desc="", traffic_impact=0.0
        )
        assert r.traffic_impact >= 0.40   # dense fog = major impact

    @patch("collectors.weather_events_collector.safe_get")
    def test_fetch_zone_parses_response(self, mock_get):
        mock_get.return_value = {
            "current": {
                "time": "2026-03-26T08:00",
                "temperature_2m": 26.0,
                "precipitation": 2.5,
                "visibility": 8000,
                "wind_speed_10m": 15.0,
                "weather_code": 61,
            }
        }
        reading = self.collector.fetch_zone("centre", 22.5726, 88.3639)
        assert reading is not None
        assert reading.precipitation_mm == 2.5
        assert reading.weather_code == 61
        assert reading.traffic_impact > 0


# ── Events collector tests ────────────────────────────────────────────────────
class TestEventsCollector:

    def setup_method(self):
        from collectors.weather_events_collector import EventsCollector
        self.collector = EventsCollector()

    def test_synthetic_events_generated(self):
        events = self.collector.generate_synthetic_events()
        assert len(events) > 0
        types = {e.event_type for e in events}
        assert len(types) > 1   # more than one event type

    def test_large_event_high_severity(self):
        events = self.collector.generate_synthetic_events()
        large = [e for e in events if e.expected_attendance >= 50000]
        for e in large:
            assert e.severity == "high"

    def test_radius_scales_with_attendance(self):
        from collectors.weather_events_collector import EventsCollector
        small = EventsCollector._attendance_to_radius(500)
        large = EventsCollector._attendance_to_radius(100000)
        assert large > small


# ── Utility tests ─────────────────────────────────────────────────────────────
class TestHelpers:

    def test_utc_now_is_iso(self):
        from utils.helpers import utc_now
        ts = utc_now()
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_validate_record_passes(self):
        from utils.helpers import validate_record
        record = {"a": 1, "b": 2, "c": 3}
        assert validate_record(record, ["a", "b"]) is True

    def test_validate_record_fails_missing(self):
        from utils.helpers import validate_record
        record = {"a": 1}
        assert validate_record(record, ["a", "b"]) is False

    def test_append_csv_creates_file(self, tmp_path):
        from utils.helpers import append_csv
        path = tmp_path / "test.csv"
        append_csv({"x": 1, "y": 2}, path)
        assert path.exists()
        content = path.read_text()
        assert "x" in content   # header present
        assert "1" in content   # value present

    def test_checksum_deterministic(self):
        from utils.helpers import checksum
        assert checksum("hello") == checksum("hello")
        assert checksum("hello") != checksum("world")


# ── Integration smoke test ────────────────────────────────────────────────────
class TestPipelineSmoke:

    def test_full_simulate_no_crash(self, tmp_path, monkeypatch):
        """Run simulate mode end-to-end with no external calls."""
        import sys
        # Patch kafka as disabled
        monkeypatch.setenv("KAFKA_ENABLED", "false")

        from pipeline.orchestrator import TrafficDataPipeline
        pipeline = TrafficDataPipeline()
        # Should complete without raising
        pipeline.simulate()
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])