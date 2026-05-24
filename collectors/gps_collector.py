"""
collectors/gps_collector.py
----------------------------
Step 3 of Data Acquisition: GPS / Floating Car Data (FCD)

Data sources (in priority order)
---------------------------------
1. HERE Traffic Flow API  – real-time speed + jam factor per road segment
2. TomTom Traffic API     – alternative commercial source
3. OSMnx graph traversal  – fallback: synthetic travel times on real road network

Key concept: Floating Car Data (FCD)
-------------------------------------
FCD aggregates anonymised GPS positions from probe vehicles
(taxis, ride-share, delivery fleets) to compute actual travel
times between any two points on the road network.

Output per segment
------------------
  - current_speed_kmh   : actual measured speed
  - free_flow_speed_kmh : speed with no congestion
  - jam_factor          : 0 (free) → 10 (standstill)
  - travel_time_sec     : time to traverse the segment now
  - confidence          : data quality indicator
"""

import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

from loguru import logger

from config.settings import settings
from utils.helpers import utc_now, append_csv, safe_get, publish_to_kafka


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class RoadSegmentFlow:
    segment_id:          str
    timestamp:           str
    start_lat:           float
    start_lon:           float
    end_lat:             float
    end_lon:             float
    length_m:            float
    current_speed_kmh:   float
    free_flow_speed_kmh: float
    jam_factor:          float       # 0–10 scale
    travel_time_sec:     float
    confidence:          float = 1.0
    source:              str = "here"

    @property
    def speed_ratio(self) -> float:
        """current / free_flow. 1.0 = free flow, <0.5 = heavy congestion."""
        if self.free_flow_speed_kmh <= 0:
            return 1.0
        return min(1.0, self.current_speed_kmh / self.free_flow_speed_kmh)

    @property
    def delay_sec(self) -> float:
        """Extra travel time vs. free-flow conditions."""
        if self.free_flow_speed_kmh <= 0:
            return 0.0
        ff_time = (self.length_m / 1000) / self.free_flow_speed_kmh * 3600
        return max(0, self.travel_time_sec - ff_time)


# ── HERE Traffic API collector ────────────────────────────────────────────────
class HERECollector:
    """
    Fetches real-time traffic flow from the HERE Traffic v7 API.

    Free tier: 250,000 requests/month (sufficient for a city-scale project).
    Sign up: developer.here.com
    """

    BASE_URL = "https://data.traffic.hereapi.com/v7/flow"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_bbox(self, lat_min: float, lon_min: float,
                   lat_max: float, lon_max: float) -> List[RoadSegmentFlow]:
        """
        Fetch traffic flow for all road segments within a bounding box.
        Returns list of RoadSegmentFlow records.
        """
        params = {
            "locationReferencing": "shape",
            "in": f"bbox:{lon_min},{lat_min},{lon_max},{lat_max}",
            "apiKey": self.api_key,
        }
        data = safe_get(self.BASE_URL, params=params)
        if not data:
            logger.warning("HERE API returned no data")
            return []

        return self._parse_response(data)

    def fetch_radius(self, lat: float, lon: float,
                     radius_m: int = 5000) -> List[RoadSegmentFlow]:
        """Fetch traffic in a circle around a point (uses bbox approximation)."""
        deg_offset = radius_m / 111_000     # ~1 degree = 111 km
        return self.fetch_bbox(
            lat - deg_offset, lon - deg_offset,
            lat + deg_offset, lon + deg_offset,
        )

    def _parse_response(self, data: Dict) -> List[RoadSegmentFlow]:
        results = []
        ts = utc_now()

        for result in data.get("results", []):
            try:
                loc    = result.get("location", {})
                flow   = result.get("currentFlow", {})
                shape  = loc.get("shape", {}).get("links", [{}])[0]
                points = shape.get("points", [])

                if len(points) < 2:
                    continue

                start = points[0]
                end   = points[-1]

                # Compute approximate segment length from point count
                length_m = shape.get("length", 200)

                current_speed = float(flow.get("speed",     0))
                ff_speed      = float(flow.get("freeFlow",  current_speed or 50))
                jam_factor    = float(flow.get("jamFactor", 0))

                if ff_speed > 0:
                    travel_time = (length_m / 1000) / current_speed * 3600 \
                                  if current_speed > 0 else 9999
                else:
                    travel_time = 9999

                seg = RoadSegmentFlow(
                    segment_id          = loc.get("id", f"seg_{len(results)}"),
                    timestamp           = ts,
                    start_lat           = start.get("lat", 0),
                    start_lon           = start.get("lng", 0),
                    end_lat             = end.get("lat", 0),
                    end_lon             = end.get("lng", 0),
                    length_m            = length_m,
                    current_speed_kmh   = round(current_speed, 1),
                    free_flow_speed_kmh = round(ff_speed, 1),
                    jam_factor          = round(jam_factor, 2),
                    travel_time_sec     = round(travel_time, 1),
                    confidence          = float(flow.get("confidence", 1.0)),
                    source              = "here",
                )
                results.append(seg)
            except (KeyError, ValueError, ZeroDivisionError) as e:
                logger.debug(f"Skipping malformed HERE segment: {e}")

        logger.info(f"HERE API: parsed {len(results)} road segments")
        return results


# ── OSMnx-based synthetic FCD (no API key needed) ────────────────────────────
class OSMnxCollector:
    """
    Uses OpenStreetMap road network to generate synthetic FCD.
    Ideal for:
      - Testing without an API key
      - Building the road graph structure
      - Offline / air-gapped environments

    Also provides the road graph used by the GCN model.
    """

    def __init__(self):
        self.cfg   = settings.road_network
        self._graph = None

    def load_graph(self, place: str = None, lat: float = None,
                   lon: float = None, radius_m: int = None):
        """
        Download the driveable road network for an area.
        Caches to disk so subsequent calls are instant.
        """
        import osmnx as ox

        place    = place    or settings.city_name
        lat      = lat      or settings.latitude
        lon      = lon      or settings.longitude
        radius_m = radius_m or settings.radius_m

        cache_path = self.cfg.output_dir / "road_graph.graphml"

        if cache_path.exists():
            logger.info(f"Loading cached road graph: {cache_path}")
            self._graph = ox.load_graphml(cache_path)
        else:
            logger.info(f"Downloading road network: {place} (r={radius_m}m)…")
            self._graph = ox.graph_from_point(
                (lat, lon),
                dist=radius_m,
                network_type=self.cfg.network_type,
                simplify=self.cfg.simplify,
            )
            ox.save_graphml(self._graph, cache_path)
            logger.info(f"Road graph saved: {cache_path}")

        nodes, edges = ox.graph_to_gdfs(self._graph)
        logger.info(f"Graph: {len(nodes)} intersections, {len(edges)} road segments")
        return self._graph

    def export_graph_features(self) -> None:
        """
        Export node and edge features as CSV for the GCN model.
        Nodes = intersections, Edges = road segments.
        """
        import osmnx as ox
        import pandas as pd

        if not self._graph:
            self.load_graph()

        nodes, edges = ox.graph_to_gdfs(self._graph)

        # Node features
        node_df = nodes[["y", "x"]].rename(columns={"y": "lat", "x": "lon"})
        node_df.index.name = "node_id"
        node_df.to_csv(self.cfg.output_dir / "graph_nodes.csv")
        logger.info(f"Exported {len(node_df)} graph nodes")

        # Edge features
        edge_cols = ["length", "speed_kph", "travel_time", "lanes", "highway"]
        available = [c for c in edge_cols if c in edges.columns]
        edge_df   = edges[available].reset_index()
        edge_df.to_csv(self.cfg.output_dir / "graph_edges.csv", index=False)
        logger.info(f"Exported {len(edge_df)} graph edges")

    def generate_synthetic_flow(self) -> List[RoadSegmentFlow]:
        """
        Assign synthetic real-time speeds to each edge in the road graph.
        Useful for end-to-end pipeline testing.
        """
        import osmnx as ox
        import random

        if not self._graph:
            self.load_graph()

        results = []
        hour    = __import__("datetime").datetime.now().hour
        ts      = utc_now()

        for u, v, data in self._graph.edges(data=True):
            length_m  = float(data.get("length", 200))
            max_speed = float(data.get("speed_kph", 50) or 50)

            # Apply time-of-day congestion
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                speed_ratio = random.uniform(0.25, 0.60)
            elif 22 <= hour or hour <= 5:
                speed_ratio = random.uniform(0.80, 1.00)
            else:
                speed_ratio = random.uniform(0.55, 0.85)

            current_speed = max(5, max_speed * speed_ratio + random.gauss(0, 3))
            jam_factor    = round((1 - speed_ratio) * 10, 1)
            travel_time   = (length_m / 1000) / current_speed * 3600 \
                            if current_speed > 0 else 9999

            # Get node coordinates
            u_data = self._graph.nodes[u]
            v_data = self._graph.nodes[v]

            seg = RoadSegmentFlow(
                segment_id          = f"{u}_{v}",
                timestamp           = ts,
                start_lat           = u_data.get("y", 0),
                start_lon           = u_data.get("x", 0),
                end_lat             = v_data.get("y", 0),
                end_lon             = v_data.get("x", 0),
                length_m            = round(length_m, 1),
                current_speed_kmh   = round(current_speed, 1),
                free_flow_speed_kmh = round(max_speed, 1),
                jam_factor          = jam_factor,
                travel_time_sec     = round(travel_time, 1),
                confidence          = 0.8,
                source              = "synthetic_osm",
            )
            results.append(seg)

        logger.info(f"Generated synthetic flow for {len(results)} road segments")
        return results


# ── Master GPS collector ──────────────────────────────────────────────────────
class GPSCollector:
    """
    Orchestrates GPS/FCD data acquisition.
    Uses HERE API if key is available, falls back to OSMnx synthetic data.
    """

    def __init__(self):
        self.cfg     = settings.gps
        self._output = self.cfg.output_dir / "road_flow.csv"
        self._osm    = OSMnxCollector()

        if settings.here_api_key:
            self._here = HERECollector(api_key=settings.here_api_key)
            logger.info("GPS collector: using HERE Traffic API")
        else:
            self._here = None
            logger.info("GPS collector: HERE key not found, using OSMnx synthetic")

    def collect(self) -> List[RoadSegmentFlow]:
        """Collect one round of GPS / FCD data."""
        if self._here:
            segments = self._here.fetch_radius(
                lat=settings.latitude,
                lon=settings.longitude,
                radius_m=settings.radius_m,
            )
        else:
            segments = self._osm.generate_synthetic_flow()

        for seg in segments:
            self._save(seg)

        return segments

    def build_road_graph(self):
        """
        Download + cache the road network graph.
        Must be called once before GCN training.
        """
        graph = self._osm.load_graph()
        self._osm.export_graph_features()
        return graph

    def run_polling_loop(self, stop_event: threading.Event = None) -> None:
        stop_event = stop_event or threading.Event()
        logger.info(f"GPS polling every {self.cfg.poll_interval_sec}s")
        while not stop_event.is_set():
            segments = self.collect()
            logger.info(f"GPS: collected {len(segments)} segment readings")
            stop_event.wait(self.cfg.poll_interval_sec)

    def _save(self, seg: RoadSegmentFlow) -> None:
        row = asdict(seg)
        row["speed_ratio"] = round(seg.speed_ratio, 3)
        row["delay_sec"]   = round(seg.delay_sec,   1)
        append_csv(row, self._output)

        if settings.kafka_enabled:
            publish_to_kafka(
                topic             = settings.kafka.topics["gps"],
                payload           = row,
                bootstrap_servers = settings.kafka.bootstrap_servers,
            )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["flow", "graph", "both"], default="both")
    args = parser.parse_args()

    collector = GPSCollector()

    if args.mode in ("graph", "both"):
        print("Building road network graph…")
        collector.build_road_graph()

    if args.mode in ("flow", "both"):
        print("Collecting road segment flow…")
        segs = collector.collect()
        if segs:
            print(f"\n{'Segment':<20} {'Speed':>8} {'FreeFlow':>10} {'Jam':>5}")
            print("-" * 48)
            for s in segs[:10]:
                print(f"{s.segment_id[:20]:<20} {s.current_speed_kmh:>7.1f}  "
                      f"{s.free_flow_speed_kmh:>9.1f}  {s.jam_factor:>4.1f}")
            if len(segs) > 10:
                print(f"  … and {len(segs)-10} more segments")
        print(f"\nData saved to: {collector._output}")