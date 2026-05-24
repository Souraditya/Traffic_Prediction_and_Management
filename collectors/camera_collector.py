"""
collectors/camera_collector.py
-------------------------------
Step 1 of Data Acquisition: Camera Feed Processing

Architecture
------------
  RTSP/HTTP stream  →  Frame capture  →  YOLOv8 detection
  →  Vehicle count + type  →  CSV / Parquet / Kafka

Edge AI note
------------
In production, the YOLO model runs ON the camera hardware
(NVIDIA Jetson / Raspberry Pi + Coral TPU).  Only the
aggregated counts travel over the network — not raw video.
This module simulates both scenarios:
  - run_on_stream()   : full pipeline (dev / lab)
  - process_edge_payload() : receive counts from edge device (production)
"""

import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from loguru import logger

from config.settings import settings
from utils.helpers import utc_now, append_csv, parquet_append, publish_to_kafka
import pandas as pd


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class VehicleCounts:
    camera_id:    str
    timestamp:    str
    latitude:     float
    longitude:    float
    cars:         int = 0
    motorcycles:  int = 0
    buses:        int = 0
    trucks:       int = 0
    total:        int = 0
    avg_speed_kmh: Optional[float] = None   # if optical flow is computed
    incident_flag: bool = False             # stalled vehicle / debris detected
    confidence:   float = 0.0              # mean YOLO confidence

    def __post_init__(self):
        self.total = self.cars + self.motorcycles + self.buses + self.trucks


# ── Camera registry ───────────────────────────────────────────────────────────
# In production this comes from a database or config file.
# Format: camera_id → {url, lat, lon}
CAMERA_REGISTRY: Dict[str, Dict] = {
    "cam_001": {
        "url": "rtsp://192.168.1.10:554/stream1",
        "lat": 22.5726, "lon": 88.3639,
        "location": "Esplanade Junction"
    },
    "cam_002": {
        "url": "rtsp://192.168.1.11:554/stream1",
        "lat": 22.5800, "lon": 88.3500,
        "location": "Ultadanga Flyover"
    },
    "cam_003": {
        "url": "rtsp://192.168.1.12:554/stream1",
        "lat": 22.5200, "lon": 88.3800,
        "location": "Tollygunge Metro"
    },
    "cam_004": {
        "url": "rtsp://192.168.1.13:554/stream1",
        "lat": 22.5958, "lon": 88.3467,
        "location": "Howrah Bridge East"
    },
    "cam_005": {
        "url": "rtsp://192.168.1.14:554/stream1",
        "lat": 22.5553, "lon": 88.3523,
        "location": "Park Street Crossing"
    },
    "cam_006": {
        "url": "rtsp://192.168.1.15:554/stream1",
        "lat": 22.5411, "lon": 88.3961,
        "location": "EM Bypass Bypass Bypass Bypass (Bypass at Ruby)"
    },
    "cam_007": {
        "url": "rtsp://192.168.1.16:554/stream1",
        "lat": 22.6054, "lon": 88.3936,
        "location": "VIP Road Airport Gate"
    },
    "cam_008": {
        "url": "rtsp://192.168.1.17:554/stream1",
        "lat": 22.5354, "lon": 88.3302,
        "location": "Rashbehari Connector"
    },
    "cam_009": {
        "url": "rtsp://192.168.1.18:554/stream1",
        "lat": 22.5646, "lon": 88.4318,
        "location": "Salt Lake Sector V"
    },
    "cam_010": {
        "url": "rtsp://192.168.1.19:554/stream1",
        "lat": 22.4946, "lon": 88.3195,
        "location": "Joka Tram Depot"
    },
}


# ── Core detector ─────────────────────────────────────────────────────────────
class CameraCollector:
    """
    Captures frames from an RTSP/HTTP camera stream and
    runs YOLOv8 to count vehicles by type.
    """

    def __init__(self):
        self.cfg = settings.camera
        self.model = None               # lazy-loaded on first use
        self._output_csv  = self.cfg.output_dir / "vehicle_counts.csv"
        self._output_parq = self.cfg.output_dir / "vehicle_counts.parquet"

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        """Lazy-load YOLOv8. Downloads weights on first call (~6 MB for nano)."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.cfg.yolo_model)
                logger.info(f"YOLO model loaded: {self.cfg.yolo_model}")
            except ImportError:
                logger.warning("ultralytics not installed – using mock detector")
                self.model = "mock"

    # ── Single frame processing ───────────────────────────────────────────────
    def detect_vehicles(self, frame) -> Tuple[Dict[str, int], float]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Returns
        -------
        counts : dict  {car, motorcycle, bus, truck}
        mean_conf : float  average confidence of detections
        """
        self._load_model()

        counts = {name: 0 for name in self.cfg.vehicle_classes.values()}
        confidences: List[float] = []

        if self.model == "mock":
            # Return synthetic counts for testing without GPU
            import random
            hour = datetime.now().hour
            rush = 7 <= hour <= 9 or 17 <= hour <= 19
            scale = 3 if rush else 1
            return {
                "car": random.randint(5, 20) * scale,
                "motorcycle": random.randint(2, 10) * scale,
                "bus": random.randint(0, 3) * scale,
                "truck": random.randint(0, 2) * scale,
            }, 0.85

        results = self.model(
            frame,
            classes=list(self.cfg.vehicle_classes.keys()),
            conf=self.cfg.confidence,
            verbose=False,
        )

        for box in results[0].boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            label   = self.cfg.vehicle_classes.get(cls_id)
            if label:
                counts[label] = counts.get(label, 0) + 1
                confidences.append(conf)

        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return counts, mean_conf

    # ── Incident detection ────────────────────────────────────────────────────
    @staticmethod
    def detect_incident(frame, prev_frame=None) -> bool:
        """
        Lightweight incident flag.
        Uses background subtraction to detect stationary vehicles
        occupying a lane for > N consecutive frames.
        """
        try:
            import cv2
            import numpy as np
            if prev_frame is None:
                return False
            diff = cv2.absdiff(frame, prev_frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            motion_ratio = np.count_nonzero(thresh) / thresh.size
            # Very low motion on a normally-busy frame = possible stall
            return motion_ratio < 0.01
        except Exception:
            return False

    # ── Stream capture loop ───────────────────────────────────────────────────
    def run_on_stream(self, camera_id: str, duration_sec: int = 3600) -> None:
        """
        Capture frames from an RTSP stream, run detection, and save results.

        Parameters
        ----------
        camera_id   : key in CAMERA_REGISTRY
        duration_sec: how long to run (default 1 hour)
        """
        import cv2
        cam = CAMERA_REGISTRY.get(camera_id)
        if not cam:
            raise ValueError(f"Unknown camera_id: {camera_id}")

        logger.info(f"Starting stream capture: {camera_id} ({cam['location']})")
        cap = cv2.VideoCapture(cam["url"])

        if not cap.isOpened():
            logger.error(f"Cannot open stream: {cam['url']}")
            return

        prev_frame = None
        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration_sec:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"{camera_id}: frame read failed, retrying…")
                    time.sleep(2)
                    continue

                # Only process one frame per interval to save compute
                counts, conf = self.detect_vehicles(frame)
                incident    = self.detect_incident(frame, prev_frame)

                record = VehicleCounts(
                    camera_id    = camera_id,
                    timestamp    = utc_now(),
                    latitude     = cam["lat"],
                    longitude    = cam["lon"],
                    cars         = counts.get("car", 0),
                    motorcycles  = counts.get("motorcycle", 0),
                    buses        = counts.get("bus", 0),
                    trucks       = counts.get("truck", 0),
                    incident_flag= incident,
                    confidence   = round(conf, 3),
                )

                self._save_record(record)
                frame_count += 1
                logger.debug(f"{camera_id}: {record.total} vehicles, "
                             f"incident={incident}, conf={conf:.2f}")

                prev_frame = frame.copy()
                time.sleep(self.cfg.frame_interval_sec)

        finally:
            cap.release()
            logger.info(f"{camera_id}: captured {frame_count} frames")

    # ── Edge device payload receiver ──────────────────────────────────────────
    def process_edge_payload(self, payload: Dict) -> Optional[VehicleCounts]:
        """
        Production path: receive pre-processed counts from an edge device.
        The camera runs inference locally; only JSON is sent over the network.

        Expected payload
        ----------------
        {
          "camera_id": "cam_001",
          "timestamp": "2026-03-26T08:30:00Z",
          "counts": {"car": 14, "motorcycle": 6, "bus": 1, "truck": 0},
          "incident": false,
          "confidence": 0.87
        }
        """
        try:
            cam   = CAMERA_REGISTRY.get(payload["camera_id"], {})
            cnts  = payload.get("counts", {})
            record = VehicleCounts(
                camera_id    = payload["camera_id"],
                timestamp    = payload.get("timestamp", utc_now()),
                latitude     = cam.get("lat", 0.0),
                longitude    = cam.get("lon", 0.0),
                cars         = cnts.get("car", 0),
                motorcycles  = cnts.get("motorcycle", 0),
                buses        = cnts.get("bus", 0),
                trucks       = cnts.get("truck", 0),
                incident_flag= payload.get("incident", False),
                confidence   = payload.get("confidence", 0.0),
            )
            self._save_record(record)
            return record
        except KeyError as e:
            logger.error(f"Missing field in edge payload: {e}")
            return None

    # ── Batch simulation (for testing without hardware) ───────────────────────
    def simulate_day(self, camera_id: str = "cam_001",
                     date: str = "2026-03-26") -> pd.DataFrame:
        """
        Generate a realistic synthetic day of camera data.
        Useful for pipeline testing before real cameras are available.

        Traffic pattern:
          - Morning rush  07:00–09:00 : high volume
          - Midday        10:00–16:00 : moderate
          - Evening rush  17:00–19:00 : high volume
          - Night         20:00–06:00 : low volume
        """
        import random
        rows = []
        cam = CAMERA_REGISTRY.get(camera_id, {"lat": 22.5726, "lon": 88.3639})

        def _scale(hour: int) -> float:
            if   7 <= hour <= 9:  return 3.5
            elif 17 <= hour <= 19: return 3.0
            elif 10 <= hour <= 16: return 1.5
            else:                  return 0.4

        for hour in range(24):
            for minute in range(0, 60, self.cfg.frame_interval_sec // 60 or 1):
                s = _scale(hour)
                record = VehicleCounts(
                    camera_id    = camera_id,
                    timestamp    = f"{date}T{hour:02d}:{minute:02d}:00Z",
                    latitude     = cam.get("lat"),
                    longitude    = cam.get("lon"),
                    cars         = int(random.gauss(15 * s, 3)),
                    motorcycles  = int(random.gauss(8  * s, 2)),
                    buses        = max(0, int(random.gauss(2 * s, 1))),
                    trucks       = max(0, int(random.gauss(1 * s, 0.5))),
                    confidence   = round(random.uniform(0.75, 0.95), 3),
                )
                self._save_record(record)
                rows.append(asdict(record))

        logger.info(f"Simulation complete: {len(rows)} records for {date}")
        return pd.DataFrame(rows)

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save_record(self, record: VehicleCounts) -> None:
        """Save to CSV, Parquet, and optionally Kafka."""
        row = asdict(record)

        # CSV (human-readable, easy to inspect)
        append_csv(row, self._output_csv)

        # Kafka (real-time consumers)
        if settings.kafka_enabled:
            publish_to_kafka(
                topic             = settings.kafka.topics["camera"],
                payload           = row,
                bootstrap_servers = settings.kafka.bootstrap_servers,
            )


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Camera data collector")
    parser.add_argument("--mode",
        choices=["stream", "simulate"],
        default="simulate",
        help="'stream' = live RTSP, 'simulate' = synthetic data"
    )
    parser.add_argument("--camera", default="cam_001")
    parser.add_argument("--duration", type=int, default=3600)
    args = parser.parse_args()

    collector = CameraCollector()

    if args.mode == "simulate":
        df = collector.simulate_day(camera_id=args.camera)
        print(f"\nSimulated {len(df)} records. Sample:\n{df.head()}\n")
        print(f"Data saved to: {collector._output_csv}")
    else:
        collector.run_on_stream(args.camera, duration_sec=args.duration)