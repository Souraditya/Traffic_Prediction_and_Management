"""
collectors/sensor_collector.py
-------------------------------
Step 2 of Data Acquisition: IoT Sensor Data

Supports two integration methods:
  A) MQTT Subscribe  – sensors push data to a broker (preferred)
  B) REST Polling    – poll a traffic management API every N seconds

Data from inductive loop sensors:
  - Speed (km/h)  : average speed of passing vehicles
  - Occupancy (%) : % of time a vehicle is present over the sensor
  - Flow (veh/h)  : vehicles per hour
  - Headway (s)   : average time gap between vehicles
"""

import time
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, asdict

from loguru import logger

from config.settings import settings
from utils.helpers import utc_now, append_csv, safe_get, publish_to_kafka


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class SensorReading:
    sensor_id:      str
    timestamp:      str
    latitude:       float
    longitude:      float
    road_segment:   str
    speed_kmh:      float           # average speed
    occupancy_pct:  float           # 0.0 – 1.0
    flow_veh_per_hr: int            # vehicles per hour
    headway_sec:    Optional[float] = None
    sensor_type:    str = "inductive_loop"   # or "infrared", "radar"
    quality_flag:   int = 1         # 1=good, 0=suspect, -1=faulty

    # Derived: congestion level based on speed thresholds (km/h)
    @property
    def congestion_level(self) -> str:
        if   self.speed_kmh < 10:  return "gridlock"
        elif self.speed_kmh < 25:  return "heavy"
        elif self.speed_kmh < 45:  return "moderate"
        elif self.speed_kmh < 65:  return "light"
        else:                      return "free_flow"


# ── Sensor registry ───────────────────────────────────────────────────────────
SENSOR_REGISTRY: Dict[str, Dict] = {
    "loop_001": {
        "lat": 22.5726, "lon": 88.3639,
        "segment": "AJC_Bose_Rd_N", "lanes": 3
    },
    "loop_002": {
        "lat": 22.5800, "lon": 88.3500,
        "segment": "VIP_Rd_E",       "lanes": 4
    },
    "loop_003": {
        "lat": 22.5200, "lon": 88.3800,
        "segment": "EM_Bypass_S",    "lanes": 4
    },
    "loop_004": {
        "lat": 22.5650, "lon": 88.3750,
        "segment": "Park_St_W",      "lanes": 2
    },
    "ir_001": {
        "lat": 22.5900, "lon": 88.3400,
        "segment": "Ultadanga_Connector", "lanes": 2,
        "type": "infrared"
    },
}


# ── MQTT subscriber ───────────────────────────────────────────────────────────
class MQTTSensorSubscriber:
    """
    Subscribe to an MQTT broker where IoT sensors publish readings.

    Topic convention:  traffic/sensors/{sensor_id}
    Payload (JSON):
    {
      "sensor_id": "loop_001",
      "ts":        1711436400000,    ← epoch milliseconds
      "speed":     42.5,
      "occupancy": 0.68,
      "flow":      1200,
      "headway":   3.0
    }
    """

    def __init__(self, on_reading: Callable[[SensorReading], None]):
        self.cfg        = settings.sensor
        self.on_reading = on_reading
        self._client    = None

    def connect(self) -> bool:
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(client_id="traffic_collector")
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.on_disconnect = self._on_disconnect

            self._client.connect(self.cfg.mqtt_broker, self.cfg.mqtt_port, keepalive=60)
            self._client.loop_start()
            logger.info(f"MQTT connected: {self.cfg.mqtt_broker}:{self.cfg.mqtt_port}")
            return True
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self.cfg.mqtt_topic)
            logger.info(f"Subscribed to: {self.cfg.mqtt_topic}")
        else:
            logger.error(f"MQTT connect error code: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning("MQTT disconnected unexpectedly, reconnecting…")
            time.sleep(5)
            client.reconnect()

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            reading = self._parse_payload(payload)
            if reading:
                self.on_reading(reading)
        except json.JSONDecodeError as e:
            logger.error(f"MQTT JSON parse error: {e}")
        except Exception as e:
            logger.error(f"MQTT message handler error: {e}")

    def _parse_payload(self, payload: Dict) -> Optional[SensorReading]:
        sensor_id = payload.get("sensor_id", "")
        meta = SENSOR_REGISTRY.get(sensor_id, {})

        # Convert epoch-ms timestamp to ISO-8601
        ts_ms = payload.get("ts", 0)
        ts    = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat() \
                if ts_ms else utc_now()

        speed    = float(payload.get("speed",     0))
        occupancy= float(payload.get("occupancy", 0))
        flow     = int(payload.get("flow",        0))

        # Basic sanity check
        if not (0 <= speed <= 200 and 0 <= occupancy <= 1 and 0 <= flow <= 10000):
            logger.warning(f"Out-of-range values for {sensor_id}: {payload}")
            quality = 0
        else:
            quality = 1

        return SensorReading(
            sensor_id      = sensor_id,
            timestamp      = ts,
            latitude       = meta.get("lat", 0.0),
            longitude      = meta.get("lon", 0.0),
            road_segment   = meta.get("segment", "unknown"),
            speed_kmh      = speed,
            occupancy_pct  = occupancy,
            flow_veh_per_hr= flow,
            headway_sec    = payload.get("headway"),
            sensor_type    = meta.get("type", "inductive_loop"),
            quality_flag   = quality,
        )

    def stop(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()


# ── REST API poller ───────────────────────────────────────────────────────────
class RESTSensorPoller:
    """
    Poll a traffic management API (e.g. city open data portal).
    Falls back to synthetic data generation for testing.
    """

    def __init__(self, api_base_url: str = "", api_key: str = ""):
        self.api_base = api_base_url
        self.api_key  = api_key
        self.cfg      = settings.sensor

    def fetch_all_sensors(self) -> List[SensorReading]:
        """Fetch latest readings for all known sensors."""
        readings = []
        for sensor_id in SENSOR_REGISTRY:
            reading = self.fetch_sensor(sensor_id)
            if reading:
                readings.append(reading)
        return readings

    def fetch_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """Fetch latest reading for one sensor from REST API."""
        if not self.api_base:
            return self._generate_synthetic(sensor_id)

        url    = f"{self.api_base}/sensors/{sensor_id}/latest"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data   = safe_get(url, headers=headers)

        if not data:
            return None

        meta = SENSOR_REGISTRY.get(sensor_id, {})
        return SensorReading(
            sensor_id       = sensor_id,
            timestamp       = data.get("timestamp", utc_now()),
            latitude        = meta.get("lat", 0.0),
            longitude       = meta.get("lon", 0.0),
            road_segment    = meta.get("segment", "unknown"),
            speed_kmh       = float(data.get("speed", 0)),
            occupancy_pct   = float(data.get("occupancy", 0)),
            flow_veh_per_hr = int(data.get("flow", 0)),
        )

    def _generate_synthetic(self, sensor_id: str) -> SensorReading:
        """
        Generate realistic synthetic sensor data.
        Used when no real API is available (offline testing).
        """
        import random
        hour = datetime.now().hour
        meta = SENSOR_REGISTRY.get(sensor_id, {})

        # Rush hour vs. off-peak profiles
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            speed     = random.gauss(25, 8)     # congested
            occupancy = random.gauss(0.75, 0.1)
            flow      = random.gauss(1400, 150)
        elif 22 <= hour or hour <= 5:
            speed     = random.gauss(70, 10)    # free flow at night
            occupancy = random.gauss(0.15, 0.05)
            flow      = random.gauss(200, 50)
        else:
            speed     = random.gauss(45, 12)    # normal
            occupancy = random.gauss(0.45, 0.1)
            flow      = random.gauss(800, 100)

        speed     = max(0, min(120, speed))
        occupancy = max(0, min(1.0, occupancy))
        flow      = max(0, int(flow))
        headway   = round(3600 / flow, 1) if flow > 0 else None

        return SensorReading(
            sensor_id       = sensor_id,
            timestamp       = utc_now(),
            latitude        = meta.get("lat", 22.5726),
            longitude       = meta.get("lon", 88.3639),
            road_segment    = meta.get("segment", "unknown"),
            speed_kmh       = round(speed, 1),
            occupancy_pct   = round(occupancy, 3),
            flow_veh_per_hr = flow,
            headway_sec     = headway,
            sensor_type     = meta.get("type", "inductive_loop"),
        )

    def run_polling_loop(self, callback: Callable[[SensorReading], None],
                         stop_event: threading.Event = None) -> None:
        """
        Continuously poll all sensors at the configured interval.
        Runs until stop_event is set (or forever if not provided).
        """
        stop_event = stop_event or threading.Event()
        logger.info(f"Starting REST polling every {self.cfg.poll_interval_sec}s")
        while not stop_event.is_set():
            readings = self.fetch_all_sensors()
            for r in readings:
                callback(r)
                logger.debug(f"  {r.sensor_id}: "
                             f"speed={r.speed_kmh:.1f} km/h, "
                             f"occ={r.occupancy_pct:.0%}, "
                             f"flow={r.flow_veh_per_hr} veh/h "
                             f"[{r.congestion_level}]")
            stop_event.wait(self.cfg.poll_interval_sec)


# ── Master collector (orchestrates both methods) ──────────────────────────────
class SensorCollector:
    """Unified sensor data collector. Tries MQTT first, falls back to REST."""

    def __init__(self):
        self.cfg     = settings.sensor
        self._output = self.cfg.output_dir / "sensor_readings.csv"

    def _on_reading(self, reading: SensorReading) -> None:
        row = asdict(reading)
        row["congestion_level"] = reading.congestion_level
        append_csv(row, self._output)

        if settings.kafka_enabled:
            publish_to_kafka(
                topic             = settings.kafka.topics["sensor"],
                payload           = row,
                bootstrap_servers = settings.kafka.bootstrap_servers,
            )

    def start_mqtt(self) -> MQTTSensorSubscriber:
        sub = MQTTSensorSubscriber(on_reading=self._on_reading)
        if sub.connect():
            return sub
        raise ConnectionError("Could not connect to MQTT broker")

    def start_rest_polling(self, api_url: str = "", api_key: str = "",
                            stop_event: threading.Event = None) -> threading.Thread:
        poller = RESTSensorPoller(api_base_url=api_url, api_key=api_key)
        thread = threading.Thread(
            target=poller.run_polling_loop,
            args=(self._on_reading, stop_event),
            daemon=True,
        )
        thread.start()
        return thread

    def collect_snapshot(self) -> List[SensorReading]:
        """Collect one reading from all sensors immediately (useful for batch jobs)."""
        poller   = RESTSensorPoller()
        readings = poller.fetch_all_sensors()
        for r in readings:
            self._on_reading(r)
        return readings


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    collector = SensorCollector()
    print("Collecting sensor snapshot (synthetic data)…\n")
    readings = collector.collect_snapshot()

    print(f"{'Sensor':<12} {'Speed':>8} {'Occ':>6} {'Flow':>8} {'Level':<12}")
    print("-" * 52)
    for r in readings:
        print(f"{r.sensor_id:<12} {r.speed_kmh:>7.1f}  "
              f"{r.occupancy_pct:>5.0%}  {r.flow_veh_per_hr:>7}  "
              f"{r.congestion_level:<12}")

    print(f"\nData saved to: {collector._output}")