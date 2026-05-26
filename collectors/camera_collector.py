"""
collectors/camera_collector.py
-------------------------------
Step 1 of Data Acquisition: Camera Feed Processing

Architecture
------------
  RTSP/HTTP stream -> Frame capture -> YOLOv8 detection
  -> ByteTrack multi-object tracking -> persistent vehicle IDs
  -> trajectory analysis -> speed, flow, occupancy -> CSV / Kafka

ByteTrack Integration (Liu et al., 2024)
-----------------------------------------
ByteTrack assigns persistent IDs to detected vehicles across frames,
enabling:
  - Trajectory-based speed estimation (displacement between frames)
  - Accurate flow counting via virtual line crossing
  - Reliable incident detection via stationary track identification
  - No double-counting when vehicles pause or are occluded

Reference:
  Liu, J., Xie, Y., Zhang, Y., & Li, H. (2024). Vehicle Flow Detection
  and Tracking Based on an Improved YOLOv8n and ByteTrack Framework.
  World Electric Vehicle Journal, 16(1), 13.
  https://doi.org/10.3390/wevj16010013

Edge AI note
------------
In production, YOLOv8 + ByteTrack runs ON the camera hardware
(NVIDIA Jetson / Raspberry Pi + Coral TPU). Only aggregated counts
travel over the network - not raw video.
"""

import time
from collections import defaultdict
from math import sqrt
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from loguru import logger

from config.settings import settings
from utils.helpers import utc_now, append_csv, parquet_append, publish_to_kafka
import pandas as pd


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class VehicleCounts:
    camera_id:        str
    timestamp:        str
    latitude:         float
    longitude:        float
    cars:             int   = 0
    motorcycles:      int   = 0
    buses:            int   = 0
    trucks:           int   = 0
    total:            int   = 0
    avg_speed_kmh:    float = 0.0   # trajectory-based speed (ByteTrack)
    flow_rate_veh_hr: float = 0.0   # virtual line crossings per hour
    incident_flag:    bool  = False  # stationary track detected
    incident_count:   int   = 0     # number of stationary tracks
    confidence:       float = 0.0   # mean YOLO confidence
    active_tracks:    int   = 0     # number of tracked vehicles in frame

    def __post_init__(self):
        self.total = self.cars + self.motorcycles + self.buses + self.trucks


# -----------------------------------------------------------------------------
# Camera registry — 10 Kolkata locations
# -----------------------------------------------------------------------------

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
        "location": "EM Bypass Ruby"
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

# Camera calibration: pixels per metre (approximate, camera-dependent)
# Used to convert pixel displacement to real-world speed
PIXELS_PER_METRE = 8.0   # tune per camera in production
FPS              = 25.0  # frames per second of the stream

# Stationary threshold: track still for this many frames = incident
STATIONARY_FRAMES_THRESHOLD = 75   # ~3 seconds at 25 FPS


# -----------------------------------------------------------------------------
# ByteTrack state per camera
# -----------------------------------------------------------------------------

@dataclass
class TrackState:
    """Per-camera ByteTrack state maintained across frames."""
    # track_id -> list of (cx, cy) centroids (last N frames)
    history:          Dict[int, List[Tuple[float, float]]] = field(
                          default_factory=lambda: defaultdict(list))
    # track_id -> consecutive frames where centroid barely moved
    stationary_count: Dict[int, int] = field(
                          default_factory=lambda: defaultdict(int))
    # track_id -> vehicle class label
    track_class:      Dict[int, str] = field(
                          default_factory=dict)
    # number of vehicles that crossed the virtual counting line this window
    line_crosses:     int  = 0
    # cumulative cross count for flow rate calculation
    window_start_time: float = field(default_factory=time.time)

    def reset_window(self):
        self.line_crosses      = 0
        self.window_start_time = time.time()


# -----------------------------------------------------------------------------
# Core detector
# -----------------------------------------------------------------------------

class CameraCollector:
    """
    Captures frames from an RTSP/HTTP camera stream, runs YOLOv8 detection
    and ByteTrack multi-object tracking to produce per-minute traffic metrics.
    """

    def __init__(self):
        self.cfg          = settings.camera
        self.model        = None
        self._output_csv  = self.cfg.output_dir / "vehicle_counts.csv"
        self._output_parq = self.cfg.output_dir / "vehicle_counts.parquet"
        # Per-camera tracking state
        self._track_states: Dict[str, TrackState] = {}

    def _get_track_state(self, camera_id: str) -> TrackState:
        if camera_id not in self._track_states:
            self._track_states[camera_id] = TrackState()
        return self._track_states[camera_id]

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        """Lazy-load YOLOv8. Downloads weights on first call (~6 MB for nano)."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.cfg.yolo_model)
                logger.info(f"YOLO model loaded: {self.cfg.yolo_model}")
            except ImportError:
                logger.warning("ultralytics not installed - using mock detector")
                self.model = "mock"

    # ── ByteTrack detection ───────────────────────────────────────────────────
    def detect_and_track(
        self,
        frame,
        camera_id: str,
        frame_height: int = 720,
    ) -> Tuple[Dict[str, int], float, float, bool, int]:
        """
        Run YOLOv8 + ByteTrack on a single BGR frame.

        ByteTrack assigns persistent IDs to each detected vehicle across
        frames, enabling trajectory-based speed and flow measurement.

        Parameters
        ----------
        frame        : BGR numpy array
        camera_id    : str — used to retrieve per-camera track state
        frame_height : int — used to define virtual counting line position

        Returns
        -------
        counts           : dict {car, motorcycle, bus, truck}
        mean_conf        : float — average YOLO detection confidence
        avg_speed_kmh    : float — trajectory-based speed estimate
        incident_flag    : bool  — True if stationary track detected
        incident_count   : int   — number of stationary tracks
        """
        self._load_model()

        if self.model == "mock":
            return self._mock_track(camera_id)

        # Virtual counting line at mid-frame height
        counting_line_y = frame_height // 2

        try:
            # model.track() runs YOLOv8 detection + ByteTrack in one call
            results = self.model.track(
                frame,
                classes   = list(self.cfg.vehicle_classes.keys()),
                conf      = self.cfg.confidence,
                tracker   = "bytetrack.yaml",  # built into ultralytics
                persist   = True,               # maintain state across frames
                verbose   = False,
            )
        except Exception as e:
            logger.warning(f"ByteTrack failed ({e}), falling back to detect-only")
            return self._detect_only(frame)

        state       = self._get_track_state(camera_id)
        counts      = {name: 0 for name in self.cfg.vehicle_classes.values()}
        confidences = []
        speeds      = []
        boxes       = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return counts, 0.0, 0.0, False, 0

        for box in boxes:
            # Skip if track ID not assigned (can happen on first frame)
            if box.id is None:
                continue

            track_id = int(box.id[0])
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])
            label    = self.cfg.vehicle_classes.get(cls_id)

            if not label:
                continue

            counts[label] = counts.get(label, 0) + 1
            confidences.append(conf)
            state.track_class[track_id] = label

            # Centroid of bounding box
            bbox = box.xyxy[0].tolist()
            cx   = (bbox[0] + bbox[2]) / 2
            cy   = (bbox[1] + bbox[3]) / 2

            history = state.history[track_id]
            history.append((cx, cy))

            # Keep only last 30 frames of history per track
            if len(history) > 30:
                history.pop(0)

            # ── Speed estimation from trajectory ──────────────────────────
            if len(history) >= 2:
                prev_cx, prev_cy = history[-2]
                dx = cx - prev_cx
                dy = cy - prev_cy
                pixel_disp   = sqrt(dx**2 + dy**2)          # pixels/frame
                metres_disp  = pixel_disp / PIXELS_PER_METRE # metres/frame
                speed_kmh    = metres_disp * FPS * 3.6       # km/h
                speeds.append(min(speed_kmh, 120.0))         # cap at 120 km/h

                # ── Stationary detection ──────────────────────────────────
                if pixel_disp < 2.0:  # barely moved (< 0.25m)
                    state.stationary_count[track_id] += 1
                else:
                    state.stationary_count[track_id] = 0

                # ── Virtual line crossing (flow count) ────────────────────
                if prev_cy < counting_line_y <= cy:
                    state.line_crosses += 1

        # ── Aggregate metrics ─────────────────────────────────────────────
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Incident: any track stationary for > threshold frames
        stationary_tracks = [
            tid for tid, cnt in state.stationary_count.items()
            if cnt >= STATIONARY_FRAMES_THRESHOLD
        ]
        incident_flag  = len(stationary_tracks) > 0
        incident_count = len(stationary_tracks)

        return counts, mean_conf, avg_speed, incident_flag, incident_count

    def _mock_track(self, camera_id: str) -> Tuple[Dict, float, float, bool, int]:
        """Synthetic ByteTrack output for testing without GPU/camera."""
        import random
        hour    = datetime.now().hour
        rush    = 7 <= hour <= 9 or 17 <= hour <= 19
        scale   = 3 if rush else 1
        # Speed is lower during rush hour (congestion effect)
        speed   = random.uniform(8, 20) if rush else random.uniform(25, 55)
        # Incident probability ~5% during rush, ~1% otherwise
        incident_prob  = 0.05 if rush else 0.01
        incident_flag  = random.random() < incident_prob
        incident_count = random.randint(1, 2) if incident_flag else 0
        counts = {
            "car":        random.randint(5, 20) * scale,
            "motorcycle": random.randint(2, 10) * scale,
            "bus":        random.randint(0, 3)  * scale,
            "truck":      random.randint(0, 2)  * scale,
        }
        return counts, 0.85, round(speed, 1), incident_flag, incident_count

    def _detect_only(self, frame) -> Tuple[Dict, float, float, bool, int]:
        """Fallback: YOLOv8 detection without tracking."""
        counts      = {name: 0 for name in self.cfg.vehicle_classes.values()}
        confidences = []
        results     = self.model(
            frame,
            classes = list(self.cfg.vehicle_classes.keys()),
            conf    = self.cfg.confidence,
            verbose = False,
        )
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = self.cfg.vehicle_classes.get(cls_id)
            if label:
                counts[label] = counts.get(label, 0) + 1
                confidences.append(conf)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return counts, mean_conf, 0.0, False, 0

    # ── Legacy single-frame detect (kept for backward compatibility) ──────────
    def detect_vehicles(self, frame) -> Tuple[Dict[str, int], float]:
        """
        Legacy method: YOLOv8 detection without ByteTrack.
        Use detect_and_track() for full tracking pipeline.
        """
        counts, conf, _, _, _ = self.detect_and_track(frame, "legacy")
        return counts, conf

    # ── Flow rate calculation ─────────────────────────────────────────────────
    def _compute_flow_rate(self, camera_id: str) -> float:
        """
        Calculate vehicles per hour from virtual line crossings
        in the current time window.
        """
        state    = self._get_track_state(camera_id)
        elapsed  = time.time() - state.window_start_time
        if elapsed < 1.0:
            return 0.0
        flow_rate = (state.line_crosses / elapsed) * 3600
        return round(flow_rate, 1)

    # ── Stream capture loop ───────────────────────────────────────────────────
    def run_on_stream(self, camera_id: str, duration_sec: int = 3600) -> None:
        """
        Capture frames from an RTSP stream, run YOLOv8 + ByteTrack,
        and save aggregated results every minute.
        """
        import cv2
        cam = CAMERA_REGISTRY.get(camera_id)
        if not cam:
            raise ValueError(f"Unknown camera_id: {camera_id}")

        logger.info(f"Starting stream: {camera_id} ({cam['location']})")
        cap         = cv2.VideoCapture(cam["url"])
        frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        if not cap.isOpened():
            logger.error(f"Cannot open stream: {cam['url']}")
            return

        start_time  = time.time()
        frame_count = 0
        state       = self._get_track_state(camera_id)

        try:
            while time.time() - start_time < duration_sec:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"{camera_id}: frame read failed, retrying")
                    time.sleep(2)
                    continue

                counts, conf, avg_speed, incident_flag, incident_count = \
                    self.detect_and_track(frame, camera_id, frame_h)

                flow_rate = self._compute_flow_rate(camera_id)

                record = VehicleCounts(
                    camera_id        = camera_id,
                    timestamp        = utc_now(),
                    latitude         = cam["lat"],
                    longitude        = cam["lon"],
                    cars             = counts.get("car", 0),
                    motorcycles      = counts.get("motorcycle", 0),
                    buses            = counts.get("bus", 0),
                    trucks           = counts.get("truck", 0),
                    avg_speed_kmh    = avg_speed,
                    flow_rate_veh_hr = flow_rate,
                    incident_flag    = incident_flag,
                    incident_count   = incident_count,
                    confidence       = round(conf, 3),
                    active_tracks    = sum(
                        1 for h in state.history.values() if len(h) > 0
                    ),
                )

                self._save_record(record)
                frame_count += 1

                logger.debug(
                    f"{camera_id}: {record.total} vehicles | "
                    f"speed={avg_speed:.1f} km/h | "
                    f"flow={flow_rate:.0f} veh/hr | "
                    f"incident={incident_flag}"
                )

                # Reset flow window every minute
                if time.time() - state.window_start_time >= 60:
                    state.reset_window()

                time.sleep(self.cfg.frame_interval_sec)

        finally:
            cap.release()
            logger.info(f"{camera_id}: captured {frame_count} frames")

    # ── Edge device payload receiver ──────────────────────────────────────────
    def process_edge_payload(self, payload: Dict) -> Optional[VehicleCounts]:
        """
        Production path: receive pre-processed counts from an edge device.
        Edge device runs YOLOv8 + ByteTrack locally; only JSON is sent.

        Expected payload
        ----------------
        {
          "camera_id": "cam_001",
          "timestamp": "2026-03-26T08:30:00Z",
          "counts": {"car": 14, "motorcycle": 6, "bus": 1, "truck": 0},
          "avg_speed_kmh": 32.5,
          "flow_rate_veh_hr": 840.0,
          "incident_flag": false,
          "incident_count": 0,
          "active_tracks": 21,
          "confidence": 0.87
        }
        """
        try:
            cam   = CAMERA_REGISTRY.get(payload["camera_id"], {})
            cnts  = payload.get("counts", {})
            record = VehicleCounts(
                camera_id        = payload["camera_id"],
                timestamp        = payload.get("timestamp", utc_now()),
                latitude         = cam.get("lat", 0.0),
                longitude        = cam.get("lon", 0.0),
                cars             = cnts.get("car", 0),
                motorcycles      = cnts.get("motorcycle", 0),
                buses            = cnts.get("bus", 0),
                trucks           = cnts.get("truck", 0),
                avg_speed_kmh    = payload.get("avg_speed_kmh", 0.0),
                flow_rate_veh_hr = payload.get("flow_rate_veh_hr", 0.0),
                incident_flag    = payload.get("incident_flag", False),
                incident_count   = payload.get("incident_count", 0),
                active_tracks    = payload.get("active_tracks", 0),
                confidence       = payload.get("confidence", 0.0),
            )
            self._save_record(record)
            return record
        except KeyError as e:
            logger.error(f"Missing field in edge payload: {e}")
            return None

    # ── Batch simulation ──────────────────────────────────────────────────────
    def simulate_day(
        self,
        camera_id: str = "cam_001",
        date: str = "2026-03-26",
    ) -> pd.DataFrame:
        """
        Generate a realistic synthetic day of camera data with
        ByteTrack-style outputs (speed, flow rate, incident count).

        Traffic pattern:
          Morning rush  07:00-09:00 : high volume, low speed
          Midday        10:00-16:00 : moderate volume, moderate speed
          Evening rush  17:00-19:00 : high volume, low speed
          Night         20:00-06:00 : low volume, high speed
        """
        import random
        rows = []
        cam  = CAMERA_REGISTRY.get(camera_id, {"lat": 22.5726, "lon": 88.3639})

        def _scale(hour: int) -> Tuple[float, float, float]:
            """Returns (volume_scale, speed_kmh, incident_prob)"""
            if   7 <= hour <= 9:   return 3.5, random.uniform(8,  20), 0.06
            elif 17 <= hour <= 19: return 3.0, random.uniform(10, 22), 0.05
            elif 10 <= hour <= 16: return 1.5, random.uniform(25, 40), 0.02
            else:                   return 0.4, random.uniform(40, 60), 0.005

        interval = self.cfg.frame_interval_sec // 60 or 1

        for hour in range(24):
            for minute in range(0, 60, interval):
                s, speed, inc_prob = _scale(hour)
                incident_flag  = random.random() < inc_prob
                incident_count = random.randint(1, 2) if incident_flag else 0
                # Flow rate inversely related to congestion
                flow_rate = random.gauss(s * 200, 30)

                record = VehicleCounts(
                    camera_id        = camera_id,
                    timestamp        = f"{date}T{hour:02d}:{minute:02d}:00Z",
                    latitude         = cam.get("lat"),
                    longitude        = cam.get("lon"),
                    cars             = max(0, int(random.gauss(15 * s, 3))),
                    motorcycles      = max(0, int(random.gauss(8  * s, 2))),
                    buses            = max(0, int(random.gauss(2  * s, 1))),
                    trucks           = max(0, int(random.gauss(1  * s, 0.5))),
                    avg_speed_kmh    = round(max(3.0, speed), 1),
                    flow_rate_veh_hr = round(max(0, flow_rate), 1),
                    incident_flag    = incident_flag,
                    incident_count   = incident_count,
                    active_tracks    = max(0, int(random.gauss(20 * s, 5))),
                    confidence       = round(random.uniform(0.75, 0.95), 3),
                )
                self._save_record(record)
                rows.append(asdict(record))

        logger.info(f"Simulation complete: {len(rows)} records for {date}")
        return pd.DataFrame(rows)

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save_record(self, record: VehicleCounts) -> None:
        """Save to CSV and optionally Kafka."""
        row = asdict(record)
        append_csv(row, self._output_csv)
        if settings.kafka_enabled:
            publish_to_kafka(
                topic             = settings.kafka.topics["camera"],
                payload           = row,
                bootstrap_servers = settings.kafka.bootstrap_servers,
            )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Camera data collector with ByteTrack")
    parser.add_argument("--mode",     choices=["stream", "simulate"], default="simulate")
    parser.add_argument("--camera",   default="cam_001")
    parser.add_argument("--duration", type=int, default=3600)
    args = parser.parse_args()

    collector = CameraCollector()

    if args.mode == "simulate":
        df = collector.simulate_day(camera_id=args.camera)
        print(f"\nSimulated {len(df)} records. Sample:")
        print(df[["timestamp", "total", "avg_speed_kmh",
                  "flow_rate_veh_hr", "incident_flag", "incident_count"]].head(10))
        print(f"\nData saved to: {collector._output_csv}")
    else:
        collector.run_on_stream(args.camera, duration_sec=args.duration)