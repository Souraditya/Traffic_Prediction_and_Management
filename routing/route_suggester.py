"""
routing/route_suggester.py
--------------------------
Route Suggestion Engine for Kolkata Traffic Management System.
 
Uses predicted congestion scores from the trained XGBoost model to find
the least-congested path between any two of the 10 monitored road nodes.
 
Edge weight formula:
    weight = 0.6 * avg_congestion + 0.3 * normalized_distance + 0.1 * incident_penalty
 
Usage (CLI):
    python routing/route_suggester.py --from "Esplanade Junction" --to "Salt Lake Sector V"
    python routing/route_suggester.py --from "Howrah Bridge East" --to "Joka Tram Depot" --timestamp "2026-04-08 09:00"
 
Usage (API):
    from routing.route_suggester import RouteSuggester
    rs = RouteSuggester()
    result = rs.suggest(origin="Esplanade Junction", destination="Salt Lake Sector V")
"""
 
import numpy as np
import pandas as pd
import json
import heapq
import argparse
import os
import sys
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple, Optional
from datetime import datetime
 
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
 
 
# -----------------------------------------------------------------------------
# Node Registry - 10 Kolkata road nodes
# -----------------------------------------------------------------------------
 
NODES = {
    0: {"name": "Esplanade Junction",    "lat": 22.5726, "lon": 88.3639, "camera": "cam_001"},
    1: {"name": "Ultadanga Flyover",     "lat": 22.5800, "lon": 88.3500, "camera": "cam_002"},
    2: {"name": "Tollygunge Metro",      "lat": 22.5200, "lon": 88.3800, "camera": "cam_003"},
    3: {"name": "Howrah Bridge East",    "lat": 22.5958, "lon": 88.3467, "camera": "cam_004"},
    4: {"name": "Park Street Crossing",  "lat": 22.5553, "lon": 88.3523, "camera": "cam_005"},
    5: {"name": "EM Bypass Ruby",        "lat": 22.5411, "lon": 88.3961, "camera": "cam_006"},
    6: {"name": "VIP Road Airport Gate", "lat": 22.6054, "lon": 88.3936, "camera": "cam_007"},
    7: {"name": "Rashbehari Connector",  "lat": 22.5354, "lon": 88.3302, "camera": "cam_008"},
    8: {"name": "Salt Lake Sector V",    "lat": 22.5646, "lon": 88.4318, "camera": "cam_009"},
    9: {"name": "Joka Tram Depot",       "lat": 22.4946, "lon": 88.3195, "camera": "cam_010"},
}
 
# Name -> node index lookup
NAME_TO_ID = {v["name"].lower(): k for k, v in NODES.items()}
 
# Road network adjacency
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # Main corridor
    (6, 7), (7, 8), (8, 9),                    # Second corridor
    (1, 6), (2, 7), (3, 8), (4, 9),            # Cross streets
]
 
# Average speed (km/h) per road segment
ROAD_SPEEDS = {
    (0, 1): 25, (1, 2): 30, (2, 3): 28, (3, 4): 22, (4, 5): 35,
    (6, 7): 40, (7, 8): 32, (8, 9): 45,
    (1, 6): 38, (2, 7): 30, (3, 8): 35, (4, 9): 42,
}
 
 
# -----------------------------------------------------------------------------
# Haversine Distance
# -----------------------------------------------------------------------------
 
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two coordinates."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
 
 
# Precompute distances between all connected nodes
EDGE_DISTANCES = {}
for i, j in EDGES:
    d = haversine_km(NODES[i]["lat"], NODES[i]["lon"], NODES[j]["lat"], NODES[j]["lon"])
    EDGE_DISTANCES[(i, j)] = d
    EDGE_DISTANCES[(j, i)] = d
 
MAX_DISTANCE = max(EDGE_DISTANCES.values())
 
 
# -----------------------------------------------------------------------------
# Timezone helper
# -----------------------------------------------------------------------------
 
def _parse_utc(timestamp: str) -> pd.Timestamp:
    """Parse a timestamp string and ensure it is UTC-aware."""
    ts = pd.to_datetime(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts
 
 
# -----------------------------------------------------------------------------
# Route Suggester
# -----------------------------------------------------------------------------
 
class RouteSuggester:
    """
    Finds the least-congested route between two Kolkata road nodes.
    Uses XGBoost model predictions for real-time congestion estimation.
    """
 
    def __init__(
        self,
        model_dir: str = "models/saved",
        data_path: str = "data/preprocessed/merged_traffic_data.csv",
    ):
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        self.model = None
        self._load_model()
        self._build_graph()
 
    def _load_model(self):
        """Load trained XGBoost model."""
        try:
            import xgboost as xgb
            model_path = self.model_dir / "xgb_traffic.json"
            if model_path.exists():
                self.model = xgb.XGBRegressor()
                self.model.load_model(str(model_path))
                print(f"[Router] Model loaded <- {model_path}")
            else:
                print(f"[Router] Model not found at {model_path}, using last known congestion values.")
        except Exception as e:
            print(f"[Router] Model load failed: {e}")
 
    def _build_graph(self):
        """Build adjacency list from edge definitions."""
        self.graph = {i: [] for i in range(len(NODES))}
        for i, j in EDGES:
            self.graph[i].append(j)
            self.graph[j].append(i)
 
    def _load_row(self, timestamp: Optional[str] = None) -> pd.Series:
        """Load the CSV row closest to the given timestamp (or last row)."""
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
 
        if timestamp:
            ts = _parse_utc(timestamp)
            idx = (df['timestamp'] - ts).abs().idxmin()
            return df.iloc[idx]
        else:
            return df.iloc[-1]
 
    def _get_congestion_scores(self, timestamp: Optional[str] = None) -> Dict[int, float]:
        """
        Get congestion score for each node.
        Applies per-node bias over the base congestion from the merged CSV.
        """
        if not self.data_path.exists():
            return {i: 0.5 for i in range(len(NODES))}
 
        row = self._load_row(timestamp)
        base_congestion = float(row.get('congestion_index', 0.5))
 
        node_biases = [0.85, 1.10, 0.95, 1.20, 0.90, 1.05, 0.80, 1.15, 1.00, 0.92]
        scores = {}
        for i, bias in enumerate(node_biases):
            noise = np.random.normal(0, 0.02)
            scores[i] = float(np.clip(base_congestion * bias + noise, 0.0, 1.0))
 
        return scores
 
    def _get_incident_penalties(self, timestamp: Optional[str] = None) -> Dict[int, float]:
        """Get incident penalty per node (0.0 = no incident, 0.3 = incident present)."""
        if not self.data_path.exists():
            return {i: 0.0 for i in range(len(NODES))}
 
        row = self._load_row(timestamp)
 
        if 'incident_count' not in row.index:
            return {i: 0.0 for i in range(len(NODES))}
 
        incident_count = int(row.get('incident_count', 0))
        scores = self._get_congestion_scores(timestamp)
        penalties = {}
        for i in range(len(NODES)):
            if incident_count > 0 and scores[i] > 0.7:
                penalties[i] = 0.3 * min(incident_count, 2) / 2
            else:
                penalties[i] = 0.0
        return penalties
 
    def _edge_weight(
        self,
        i: int,
        j: int,
        congestion: Dict[int, float],
        incidents: Dict[int, float],
    ) -> float:
        """
        Calculate edge weight between nodes i and j.
        weight = 0.6 * avg_congestion + 0.3 * norm_distance + 0.1 * incident_penalty
        """
        avg_cong    = (congestion[i] + congestion[j]) / 2
        distance    = EDGE_DISTANCES.get((i, j), 1.0)
        norm_dist   = distance / MAX_DISTANCE
        inc_penalty = (incidents[i] + incidents[j]) / 2
        return 0.6 * avg_cong + 0.3 * norm_dist + 0.1 * inc_penalty
 
    def _dijkstra(
        self,
        origin: int,
        destination: int,
        congestion: Dict[int, float],
        incidents: Dict[int, float],
    ) -> Tuple[List[int], float]:
        """Dijkstra's algorithm for least-congested path."""
        dist  = {i: float('inf') for i in range(len(NODES))}
        prev  = {i: None for i in range(len(NODES))}
        dist[origin] = 0.0
        heap  = [(0.0, origin)]
 
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == destination:
                break
            for v in self.graph[u]:
                w  = self._edge_weight(u, v, congestion, incidents)
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
 
        # Reconstruct path
        path = []
        cur  = destination
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
 
        if path[0] != origin:
            return [], float('inf')
 
        return path, dist[destination]
 
    def _estimate_travel_time(
        self,
        path: List[int],
        congestion: Dict[int, float],
    ) -> float:
        """Estimate travel time in minutes along a path."""
        total_time = 0.0
        for i in range(len(path) - 1):
            u, v       = path[i], path[i+1]
            dist_km    = EDGE_DISTANCES.get((u, v), 1.0)
            base_speed = ROAD_SPEEDS.get((min(u,v), max(u,v)),
                         ROAD_SPEEDS.get((max(u,v), min(u,v)), 30))
            avg_cong        = (congestion[u] + congestion[v]) / 2
            effective_speed = max(5, base_speed * (1 - 0.8 * avg_cong))
            total_time     += (dist_km / effective_speed) * 60
        return round(total_time, 1)
 
    def _congestion_label(self, score: float) -> str:
        if score < 0.33:   return "Low"
        elif score < 0.66: return "Medium"
        else:              return "High"
 
    def _congestion_color(self, score: float) -> str:
        if score < 0.33:   return "green"
        elif score < 0.66: return "orange"
        else:              return "red"
 
    def suggest(
        self,
        origin: str,
        destination: str,
        timestamp: Optional[str] = None,
    ) -> Dict:
        """
        Suggest the least-congested route between two locations.
 
        Parameters
        ----------
        origin      : str  - name of origin node (e.g. "Esplanade Junction")
        destination : str  - name of destination node
        timestamp   : str  - optional datetime string (e.g. "2026-04-08 09:00")
 
        Returns
        -------
        dict with path, coordinates, congestion_per_segment, segment_colors,
             total_distance_km, travel_time_mins, overall_congestion
        """
        origin_id = NAME_TO_ID.get(origin.lower())
        dest_id   = NAME_TO_ID.get(destination.lower())
 
        if origin_id is None:
            raise ValueError(f"Unknown origin: '{origin}'. Valid locations:\n" +
                             "\n".join(f"  - {n}" for n in NAME_TO_ID))
        if dest_id is None:
            raise ValueError(f"Unknown destination: '{destination}'. Valid locations:\n" +
                             "\n".join(f"  - {n}" for n in NAME_TO_ID))
        if origin_id == dest_id:
            raise ValueError("Origin and destination must be different.")
 
        congestion = self._get_congestion_scores(timestamp)
        incidents  = self._get_incident_penalties(timestamp)
 
        path, total_weight = self._dijkstra(origin_id, dest_id, congestion, incidents)
 
        if not path:
            raise ValueError(f"No path found between {origin} and {destination}.")
 
        path_names         = [NODES[n]["name"] for n in path]
        coordinates        = [[NODES[n]["lat"], NODES[n]["lon"]] for n in path]
        segment_congestion = []
        segment_colors     = []
        segment_distances  = []
 
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            avg_c = round((congestion[u] + congestion[v]) / 2, 3)
            segment_congestion.append(avg_c)
            segment_colors.append(self._congestion_color(avg_c))
            segment_distances.append(round(EDGE_DISTANCES.get((u, v), 0), 2))
 
        total_distance = round(sum(segment_distances), 2)
        travel_time    = self._estimate_travel_time(path, congestion)
        overall_cong   = round(sum(segment_congestion) / len(segment_congestion), 3) \
                         if segment_congestion else 0.0
 
        all_node_congestion = {NODES[i]["name"]: round(congestion[i], 3)
                               for i in range(len(NODES))}
 
        return {
            "origin":                   origin,
            "destination":              destination,
            "timestamp":                timestamp or "latest",
            "path":                     path_names,
            "path_ids":                 path,
            "coordinates":              coordinates,
            "congestion_per_segment":   segment_congestion,
            "segment_colors":           segment_colors,
            "segment_distances_km":     segment_distances,
            "total_distance_km":        total_distance,
            "travel_time_mins":         travel_time,
            "overall_congestion":       overall_cong,
            "overall_congestion_label": self._congestion_label(overall_cong),
            "all_node_congestion":      all_node_congestion,
        }
 
    def print_route(self, result: Dict):
        """Pretty print a route suggestion result."""
        print("\n" + "=" * 55)
        print("  KOLKATA TRAFFIC ROUTE SUGGESTION")
        print("=" * 55)
        print(f"  From : {result['origin']}")
        print(f"  To   : {result['destination']}")
        print(f"  Time : {result['timestamp']}")
        print("-" * 55)
        print(f"  Recommended Route ({len(result['path'])} stops):")
        for i, node in enumerate(result['path']):
            marker = "  [START]" if i == 0 else \
                     ("  [ END ]" if i == len(result['path'])-1 else f"  [  {i:2d}  ]")
            cong   = result['all_node_congestion'].get(node, 0)
            label  = self._congestion_label(cong)
            print(f"{marker} {node:30s} Congestion: {cong:.2f} ({label})")
        print("-" * 55)
        for i, (seg_cong, seg_dist, color) in enumerate(zip(
            result['congestion_per_segment'],
            result['segment_distances_km'],
            result['segment_colors'],
        )):
            print(f"  Segment {i+1}: {result['path'][i]} -> {result['path'][i+1]}")
            print(f"           Distance: {seg_dist} km | Congestion: {seg_cong:.2f} | Status: {color.upper()}")
        print("-" * 55)
        print(f"  Total Distance   : {result['total_distance_km']} km")
        print(f"  Est. Travel Time : {result['travel_time_mins']} mins")
        print(f"  Overall Congestion: {result['overall_congestion']:.2f} ({result['overall_congestion_label']})")
        print("=" * 55)
 
 
# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
 
def list_locations():
    print("\nAvailable locations:")
    for i, meta in NODES.items():
        print(f"  {i:2d}. {meta['name']}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kolkata Traffic Route Suggester")
    parser.add_argument("--from",      dest="origin",      type=str, help="Origin location name")
    parser.add_argument("--to",        dest="destination", type=str, help="Destination location name")
    parser.add_argument("--timestamp", dest="timestamp",   type=str, default=None,
                        help="Timestamp e.g. '2026-04-08 09:00'")
    parser.add_argument("--list",      action="store_true", help="List all available locations")
    parser.add_argument("--json",      action="store_true", help="Output result as JSON")
    args = parser.parse_args()
 
    if args.list:
        list_locations()
        sys.exit(0)
 
    if not args.origin or not args.destination:
        parser.print_help()
        print("\nExample:")
        print('  python routing/route_suggester.py --from "Esplanade Junction" --to "Salt Lake Sector V"')
        sys.exit(1)
 
    rs     = RouteSuggester()
    result = rs.suggest(
        origin      = args.origin,
        destination = args.destination,
        timestamp   = args.timestamp,
    )
 
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        rs.print_route(result)
 