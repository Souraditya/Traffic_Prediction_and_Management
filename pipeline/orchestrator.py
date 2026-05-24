"""
pipeline/orchestrator.py
-------------------------
Master pipeline that launches and manages all data collectors.

Modes
------
  full      : all collectors running continuously (production)
  snapshot  : one round of collection from all sources (cron / testing)
  simulate  : pure synthetic data, no external APIs needed
  build     : road graph download only
"""

import threading
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

from loguru import logger

from config.settings import settings
from utils.helpers import setup_logger
from collectors.camera_collector         import CameraCollector
from collectors.sensor_collector         import SensorCollector
from collectors.gps_collector            import GPSCollector
from collectors.weather_events_collector import WeatherCollector, EventsCollector


class TrafficDataPipeline:
    """
    Orchestrates all data acquisition threads.
    Handles graceful shutdown on SIGINT / SIGTERM.
    """

    def __init__(self):
        setup_logger()
        self._stop_event = threading.Event()
        self._threads: Dict[str, threading.Thread] = {}

        # Collectors
        self.camera   = CameraCollector()
        self.sensor   = SensorCollector()
        self.gps      = GPSCollector()
        self.weather  = WeatherCollector()
        self.events   = EventsCollector()

    # ── Startup helpers ───────────────────────────────────────────────────────
    def _spawn(self, name: str, target, *args) -> threading.Thread:
        t = threading.Thread(
            target=target, args=args, name=name, daemon=True
        )
        t.start()
        self._threads[name] = t
        logger.info(f"Thread started: {name}")
        return t

    # ── Modes ─────────────────────────────────────────────────────────────────
    def build_graph(self):
        """Download and cache the road network (run once before training)."""
        logger.info("Building road network graph…")
        graph = self.gps.build_road_graph()
        logger.success(
            f"Graph ready: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        return graph

    def snapshot(self):
        """
        Collect one round of data from all sources.
        Ideal for cron-based batch collection.
        """
        logger.info("=== Snapshot collection started ===")
        start = datetime.now()

        # Camera simulation
        for cam_id in ["cam_001", "cam_002", "cam_003"]:
            self.camera.simulate_day(camera_id=cam_id)

        # Sensor readings
        sensor_readings = self.sensor.collect_snapshot()
        logger.info(f"Sensors: {len(sensor_readings)} readings")

        # GPS / road flow
        gps_segments = self.gps.collect()
        logger.info(f"GPS: {len(gps_segments)} segments")

        # Weather
        weather = self.weather.fetch_all_zones()
        logger.info(f"Weather: {len(weather)} zones")

        # Events
        self.events.generate_synthetic_events()

        elapsed = (datetime.now() - start).total_seconds()
        logger.success(f"Snapshot complete in {elapsed:.1f}s")
        self._print_summary(sensor_readings, gps_segments, weather)

    def simulate(self):
        """
        Full synthetic simulation — no network calls, no hardware.
        Generates one day of data across all sources.
        """
        logger.info("=== Simulation mode (no API keys required) ===")
        self.snapshot()

    def run_continuous(self):
        """
        Start all collectors as background threads.
        Blocks until Ctrl+C / SIGTERM.
        """
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("=== Starting continuous collection pipeline ===")

        # Sensor polling (fastest: every 30s)
        self._spawn("sensor_poller",
            self.sensor.start_rest_polling,
            "", "", self._stop_event
        )

        # GPS polling (every 5 minutes)
        self._spawn("gps_poller",
            self.gps.run_polling_loop, self._stop_event
        )

        # Weather polling (every hour)
        self._spawn("weather_poller",
            self.weather.run_polling_loop, self._stop_event
        )

        logger.info(
            f"Pipeline running with {len(self._threads)} threads. "
            "Press Ctrl+C to stop."
        )

        # Block main thread
        self._stop_event.wait()
        logger.info("Pipeline stopped cleanly")

    def _handle_shutdown(self, signum, frame):
        logger.info(f"Shutdown signal received ({signum})")
        self._stop_event.set()

    @staticmethod
    def _print_summary(sensors, gps_segs, weather) -> None:
        print("\n" + "═" * 55)
        print("  DATA ACQUISITION SUMMARY")
        print("═" * 55)

        if sensors:
            speeds = [r.speed_kmh for r in sensors]
            avg_s  = sum(speeds) / len(speeds)
            print(f"\n  IoT Sensors")
            print(f"    Readings     : {len(sensors)}")
            print(f"    Avg speed    : {avg_s:.1f} km/h")
            cong = {r.congestion_level for r in sensors}
            print(f"    Levels seen  : {', '.join(sorted(cong))}")

        if gps_segs:
            jams = [s.jam_factor for s in gps_segs]
            avg_j = sum(jams) / len(jams)
            print(f"\n  GPS / Road Flow")
            print(f"    Segments     : {len(gps_segs)}")
            print(f"    Avg jam score: {avg_j:.2f} / 10.0")

        if weather:
            impacts = [w.traffic_impact for w in weather]
            avg_i   = sum(impacts) / len(impacts)
            print(f"\n  Weather")
            print(f"    Zones        : {len(weather)}")
            print(f"    Avg impact   : {avg_i:.0%}")
            cond = {w.weather_desc for w in weather}
            print(f"    Conditions   : {', '.join(sorted(cond))}")

        print(f"\n  Output directory : {settings.camera.output_dir.parent}")
        print("═" * 55 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Traffic data acquisition pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipeline.orchestrator --mode simulate     # no API keys needed
  python -m pipeline.orchestrator --mode snapshot     # single collection round
  python -m pipeline.orchestrator --mode build        # download road graph
  python -m pipeline.orchestrator --mode full         # continuous production mode
        """
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "snapshot", "build", "full"],
        default="simulate",
        help="Collection mode (default: simulate)"
    )
    args = parser.parse_args()

    pipeline = TrafficDataPipeline()

    if args.mode == "build":
        pipeline.build_graph()
    elif args.mode == "simulate":
        pipeline.simulate()
    elif args.mode == "snapshot":
        pipeline.snapshot()
    elif args.mode == "full":
        pipeline.run_continuous()