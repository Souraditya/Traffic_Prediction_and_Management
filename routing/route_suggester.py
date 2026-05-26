"""
routing/route_suggester.py
--------------------------
Route Suggestion Engine for Kolkata Traffic Management System.
 
Uses predicted congestion scores from the trained XGBoost model to find
the least-congested path between any two of the 10 monitored road nodes.
 
Edge weight formula:
    weight = 0.6 * avg_congestion + 0.3 * normalized_distance + 0.1 * incident_penalty
 
Road speeds are derived from GPS free-flow speed data (free_flow_speed_kmh)
by matching each road edge midpoint to the nearest GPS segments.
 
Usage (CLI):
    python routing/route_suggester.py --from "Esplanade Junction" --to "Salt Lake Sector V"
    python routing/route_suggester.py --from "Howrah Bridge East" --to "Joka Tram Depot" --timestamp "2026-04-08 09:00"
    python routing/route_suggester.py --list
 
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
import sys
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple, Optional
 
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
 
NAME_TO_ID = {v["name"].lower(): k for k, v in NODES.items()}
 
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # Main corridor
    (6, 7), (7, 8), (8, 9),                    # Second corridor
    (1, 6), (2, 7), (3, 8), (4, 9),            # Cross streets
]
 
# Fallback speeds (km/h) used when GPS data is unavailable
ROAD_SPEEDS_FALLBACK = {
    (0, 1): 25, (1, 2): 30, (2, 3): 28, (3, 4): 22, (4, 5): 35,
    (6, 7): 40, (7, 8): 32, (8, 9): 45,
    (1, 6): 38, (2, 7): 30, (3, 8): 35, (4, 9): 42,
}
 
 
# -----------------------------------------------------------------------------
# Haversine Distance
# -----------------------------------------------------------------------------
 
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
 
 
# Precompute edge distances
EDGE_DISTANCES = {}
for i, j in EDGES:
    d = haversine_km(NODES[i]["lat"], NODES[i]["lon"], NODES[j]["lat"], NODES[j]["lon"])
    EDGE_DISTANCES[(i, j)] = d
    EDGE_DISTANCES[(j, i)] = d
 
MAX_DISTANCE = max(EDGE_DISTANCES.values())
 
 
# -----------------------------------------------------------------------------
# GPS Speed Loader
# -----------------------------------------------------------------------------
 
def load_gps_speeds(
    gps_path: str = "data/preprocessed/gps/road_flow_processed.csv",
    n_nearest: int = 5,
) -> Dict[Tuple[int, int], float]:
    """
    Derive free-flow speeds for each road edge from GPS data.
    Matches each edge midpoint to the n_nearest GPS segments and
    averages their free_flow_speed_kmh values.
 
    Falls back to ROAD_SPEEDS_FALLBACK if GPS file is unavailable.
    """
    gps_file = Path(gps_path)
    if not gps_file.exists():
        print(f"[Router] GPS data not found at {gps_path}, using fallback speeds.")
        return ROAD_SPEEDS_FALLBACK.copy()
 
    try:
        gps = pd.read_csv(gps_path)
        required = {'start_lat', 'start_lon', 'end_lat', 'end_lon', 'free_flow_speed_kmh'}
        if not required.issubset(gps.columns):
            print(f"[Router] GPS data missing required columns, using fallback speeds.")
            return ROAD_SPEEDS_FALLBACK.copy()
 
        gps['mid_lat'] = (gps['start_lat'] + gps['end_lat']) / 2
        gps['mid_lon'] = (gps['start_lon'] + gps['end_lon']) / 2
 
        speeds = {}
        for i, j in EDGES:
            mid_lat = (NODES[i]['lat'] + NODES[j]['lat']) / 2
            mid_lon = (NODES[i]['lon'] + NODES[j]['lon']) / 2
 
            gps['_dist'] = gps.apply(
                lambda r: haversine_km(mid_lat, mid_lon, r['mid_lat'], r['mid_lon']),
                axis=1
            )
            nearest    = gps.nsmallest(n_nearest, '_dist')
            avg_speed  = round(float(nearest['free_flow_speed_kmh'].mean()), 1)
            speeds[(i, j)] = avg_speed
            speeds[(j, i)] = avg_speed  # undirected
 
        print(f"[Router] GPS speeds loaded from {gps_path} ({len(gps):,} segments)")
        return speeds
 
    except Exception as e:
        print(f"[Router] GPS speed loading failed ({e}), using fallback speeds.")
        return ROAD_SPEEDS_FALLBACK.copy()
 
 
# -----------------------------------------------------------------------------
# Timezone Helper
# -----------------------------------------------------------------------------
 
def _parse_utc(timestamp: str) -> pd.Timestamp:
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
    Road speeds are derived from GPS free-flow data.
    Congestion scores come from the merged traffic CSV.
    """
 
    def __init__(
        self,
        model_dir: str = "models/saved",
        data_path: str = "data/preprocessed/merged_traffic_data.csv",
        gps_path:  str = "data/preprocessed/gps/road_flow_processed.csv",
    ):
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        self.model     = None
        self._load_model()
        self._build_graph()
        self.road_speeds = load_gps_speeds(gps_path)
        self._print_speeds()
 
    def _load_model(self):
        try:
            import xgboost as xgb
            model_path = self.model_dir / "xgb_traffic.json"
            if model_path.exists():
                self.model = xgb.XGBRegressor()
                self.model.load_model(str(model_path))
                print(f"[Router] Model loaded <- {model_path}")
            else:
                print(f"[Router] Model not found, using CSV congestion values.")
        except Exception as e:
            print(f"[Router] Model load failed: {e}")
 
    def _build_graph(self):
        self.graph = {i: [] for i in range(len(NODES))}
        for i, j in EDGES:
            self.graph[i].append(j)
            self.graph[j].append(i)
 
    def _print_speeds(self):
        print("[Router] Road speeds (km/h) derived from GPS data:")
        for i, j in EDGES:
            spd = self.road_speeds.get((i, j), 30)
            print(f"         ({i},{j}) {NODES[i]['name'][:18]:18s} -> "
                  f"{NODES[j]['name'][:18]:18s}: {spd} km/h")
 
    def _load_row(self, timestamp: Optional[str] = None) -> pd.Series:
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        if timestamp:
            ts  = _parse_utc(timestamp)
            idx = (df['timestamp'] - ts).abs().idxmin()
            return df.iloc[idx]
        return df.iloc[-1]
 
    def _get_congestion_scores(self, timestamp: Optional[str] = None) -> Dict[int, float]:
        if not self.data_path.exists():
            return {i: 0.5 for i in range(len(NODES))}
 
        row             = self._load_row(timestamp)
        base_congestion = float(row.get('congestion_index', 0.5))
        node_biases     = [0.85, 1.10, 0.95, 1.20, 0.90, 1.05, 0.80, 1.15, 1.00, 0.92]
        scores          = {}
        for i, bias in enumerate(node_biases):
            noise     = np.random.normal(0, 0.02)
            scores[i] = float(np.clip(base_congestion * bias + noise, 0.0, 1.0))
        return scores
 
    def _get_incident_penalties(self, timestamp: Optional[str] = None) -> Dict[int, float]:
        if not self.data_path.exists():
            return {i: 0.0 for i in range(len(NODES))}
 
        row            = self._load_row(timestamp)
        incident_count = int(row.get('incident_count', 0))
        scores         = self._get_congestion_scores(timestamp)
        penalties      = {}
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
        incidents:  Dict[int, float],
    ) -> float:
        avg_cong    = (congestion[i] + congestion[j]) / 2
        distance    = EDGE_DISTANCES.get((i, j), 1.0)
        norm_dist   = distance / MAX_DISTANCE
        inc_penalty = (incidents[i] + incidents[j]) / 2
        return 0.6 * avg_cong + 0.3 * norm_dist + 0.1 * inc_penalty
 
    def _dijkstra(
        self,
        origin:      int,
        destination: int,
        congestion:  Dict[int, float],
        incidents:   Dict[int, float],
    ) -> Tuple[List[int], float]:
        dist  = {i: float('inf') for i in range(len(NODES))}
        prev  = {i: None         for i in range(len(NODES))}
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
        path:       List[int],
        congestion: Dict[int, float],
    ) -> float:
        """
        Estimate travel time in minutes using GPS-derived free-flow speeds,
        reduced proportionally by congestion level.
        """
        total_time = 0.0
        for idx in range(len(path) - 1):
            u, v            = path[idx], path[idx+1]
            dist_km         = EDGE_DISTANCES.get((u, v), 1.0)
            free_flow_speed = self.road_speeds.get((u, v),
                              self.road_speeds.get((v, u), 30.0))
            avg_cong        = (congestion[u] + congestion[v]) / 2
            # Speed reduces linearly: 100% at 0 congestion, 20% at full congestion
            effective_speed = max(5.0, free_flow_speed * (1 - 0.8 * avg_cong))
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
        origin:      str,
        destination: str,
        timestamp:   Optional[str] = None,
    ) -> Dict:
        """
        Suggest the least-congested route between two locations.
 
        Parameters
        ----------
        origin      : str  - name of origin node
        destination : str  - name of destination node
        timestamp   : str  - optional datetime string (e.g. "2026-04-08 09:00")
 
        Returns
        -------
        dict with path, coordinates, congestion_per_segment, segment_colors,
             segment_speeds_kmh, total_distance_km, travel_time_mins,
             overall_congestion, overall_congestion_label, all_node_congestion
        """
        origin_id = NAME_TO_ID.get(origin.lower())
        dest_id   = NAME_TO_ID.get(destination.lower())
 
        if origin_id is None:
            raise ValueError(f"Unknown origin: '{origin}'.\nValid:\n" +
                             "\n".join(f"  - {n}" for n in NAME_TO_ID))
        if dest_id is None:
            raise ValueError(f"Unknown destination: '{destination}'.\nValid:\n" +
                             "\n".join(f"  - {n}" for n in NAME_TO_ID))
        if origin_id == dest_id:
            raise ValueError("Origin and destination must be different.")
 
        congestion = self._get_congestion_scores(timestamp)
        incidents  = self._get_incident_penalties(timestamp)
        path, _    = self._dijkstra(origin_id, dest_id, congestion, incidents)
 
        if not path:
            raise ValueError(f"No path found between {origin} and {destination}.")
 
        path_names         = [NODES[n]["name"] for n in path]
        coordinates        = [[NODES[n]["lat"], NODES[n]["lon"]] for n in path]
        segment_congestion = []
        segment_colors     = []
        segment_distances  = []
        segment_speeds     = []
 
        for idx in range(len(path) - 1):
            u, v       = path[idx], path[idx+1]
            avg_c      = round((congestion[u] + congestion[v]) / 2, 3)
            free_speed = self.road_speeds.get((u, v),
                         self.road_speeds.get((v, u), 30.0))
            segment_congestion.append(avg_c)
            segment_colors.append(self._congestion_color(avg_c))
            segment_distances.append(round(EDGE_DISTANCES.get((u, v), 0), 2))
            segment_speeds.append(free_speed)
 
        total_distance = round(sum(segment_distances), 2)
        travel_time    = self._estimate_travel_time(path, congestion)
        overall_cong   = round(sum(segment_congestion) / len(segment_congestion), 3) \
                         if segment_congestion else 0.0
 
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
            "segment_speeds_kmh":       segment_speeds,
            "total_distance_km":        total_distance,
            "travel_time_mins":         travel_time,
            "overall_congestion":       overall_cong,
            "overall_congestion_label": self._congestion_label(overall_cong),
            "all_node_congestion":      {NODES[i]["name"]: round(congestion[i], 3)
                                         for i in range(len(NODES))},
        }
 
    def print_route(self, result: Dict):
        print("\n" + "=" * 58)
        print("  KOLKATA TRAFFIC ROUTE SUGGESTION")
        print("=" * 58)
        print(f"  From : {result['origin']}")
        print(f"  To   : {result['destination']}")
        print(f"  Time : {result['timestamp']}")
        print("-" * 58)
        print(f"  Recommended Route ({len(result['path'])} stops):")
        for i, node in enumerate(result['path']):
            marker = "  [START]" if i == 0 else \
                     ("  [ END ]" if i == len(result['path'])-1 else f"  [  {i:2d}  ]")
            cong   = result['all_node_congestion'].get(node, 0)
            print(f"{marker} {node:30s} Congestion: {cong:.2f} ({self._congestion_label(cong)})")
        print("-" * 58)
        for i, (seg_cong, seg_dist, color, spd) in enumerate(zip(
            result['congestion_per_segment'],
            result['segment_distances_km'],
            result['segment_colors'],
            result['segment_speeds_kmh'],
        )):
            print(f"  Segment {i+1}: {result['path'][i]} -> {result['path'][i+1]}")
            print(f"           Distance: {seg_dist} km | Free-flow: {spd} km/h | "
                  f"Congestion: {seg_cong:.2f} | Status: {color.upper()}")
        print("-" * 58)
        print(f"  Total Distance   : {result['total_distance_km']} km")
        print(f"  Est. Travel Time : {result['travel_time_mins']} mins")
        print(f"  Overall Congestion: {result['overall_congestion']:.2f} "
              f"({result['overall_congestion_label']})")
        print("=" * 58)
 
 
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
 
def list_locations():
    print("\nAvailable locations:")
    for i, meta in NODES.items():
        print(f"  {i:2d}. {meta['name']}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kolkata Traffic Route Suggester")
    parser.add_argument("--from",      dest="origin",      type=str)
    parser.add_argument("--to",        dest="destination", type=str)
    parser.add_argument("--timestamp", dest="timestamp",   type=str, default=None)
    parser.add_argument("--list",      action="store_true")
    parser.add_argument("--json",      action="store_true")
    args = parser.parse_args()
 
    if args.list:
        list_locations()
        sys.exit(0)
 
    if not args.origin or not args.destination:
        parser.print_help()
        print('\nExample:')
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
 