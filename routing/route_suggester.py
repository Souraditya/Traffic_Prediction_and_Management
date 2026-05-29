"""
routing/route_suggester.py
--------------------------
Route Suggestion Engine for Kolkata Traffic Management System.

Congestion scores are derived from an ensemble of:
  - STGCN-LSTM  : predicts congestion for all 10 nodes simultaneously
                  using the road network graph and a 12-minute history window.
  - XGBoost     : predicts aggregate congestion from 36 tabular features;
                  per-node scores are obtained by applying node-specific
                  residual corrections learned from the training data.

Ensemble formula (per node i):
    congestion[i] = alpha * stgcn[i] + (1 - alpha) * xgb_node[i]
    where alpha = 0.6  (STGCN weighted higher for spatial coherence)

Edge weight formula:
    weight = 0.6 * avg_congestion + 0.3 * normalized_distance + 0.1 * incident_penalty

Road speeds are derived from GPS free-flow speed data (free_flow_speed_kmh).

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

# Fallback speeds (km/h) when GPS data is unavailable
ROAD_SPEEDS_FALLBACK = {
    (0, 1): 25, (1, 2): 30, (2, 3): 28, (3, 4): 22, (4, 5): 35,
    (6, 7): 40, (7, 8): 32, (8, 9): 45,
    (1, 6): 38, (2, 7): 30, (3, 8): 35, (4, 9): 42,
}

# Node-specific residual corrections for XGBoost predictions.
# These reflect that some intersections are consistently busier than the
# network average. Derived from per-camera traffic profiles in the dataset.
# Values > 1.0 mean the node is typically busier than the network average.
NODE_RESIDUALS = np.array([
    0.85,  # 0 Esplanade Junction    — moderate traffic
    1.10,  # 1 Ultadanga Flyover     — above average
    0.95,  # 2 Tollygunge Metro      — moderate
    1.20,  # 3 Howrah Bridge East    — busiest node
    0.90,  # 4 Park Street Crossing  — moderate
    1.05,  # 5 EM Bypass Ruby        — slightly above average
    0.80,  # 6 VIP Road Airport Gate — lighter traffic
    1.15,  # 7 Rashbehari Connector  — above average
    1.00,  # 8 Salt Lake Sector V    — average
    0.92,  # 9 Joka Tram Depot       — slightly below average
])

# Ensemble weight for STGCN-LSTM (remainder goes to XGBoost)
STGCN_WEIGHT = 0.6
XGB_WEIGHT   = 0.4

# STGCN input parameters
STGCN_SEQ_LEN   = 12   # timesteps in input window
STGCN_N_NODES   = 10
STGCN_N_FEATURES = 6   # vehicle_count, avg_speed, congestion, rainfall, temperature, hour


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
            print("[Router] GPS data missing required columns, using fallback speeds.")
            return ROAD_SPEEDS_FALLBACK.copy()

        gps['mid_lat'] = (gps['start_lat'] + gps['end_lat']) / 2
        gps['mid_lon'] = (gps['start_lon'] + gps['end_lon']) / 2

        speeds = {}
        for i, j in EDGES:
            mid_lat = (NODES[i]['lat'] + NODES[j]['lat']) / 2
            mid_lon = (NODES[i]['lon'] + NODES[j]['lon']) / 2
            gps['_dist'] = gps.apply(
                lambda r: haversine_km(mid_lat, mid_lon, r['mid_lat'], r['mid_lon']), axis=1
            )
            nearest   = gps.nsmallest(n_nearest, '_dist')
            avg_speed = round(float(nearest['free_flow_speed_kmh'].mean()), 1)
            speeds[(i, j)] = avg_speed
            speeds[(j, i)] = avg_speed

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

    Congestion scores are produced by an ensemble of:
      - STGCN-LSTM (weight=0.6): predicts per-node congestion using the road
        network graph and a 12-minute temporal window.
      - XGBoost    (weight=0.4): predicts aggregate congestion from 36 tabular
        features; per-node scores are derived via node-specific residual
        corrections.

    If either model is unavailable, the engine gracefully falls back to the
    available model or to raw CSV values.
    """

    def __init__(
        self,
        model_dir: str = "models/saved",
        data_path: str = "data/preprocessed/merged_traffic_data.csv",
        gps_path:  str = "data/preprocessed/gps/road_flow_processed.csv",
        node_features_path: str = "data/preprocessed/node_features.npy",
        adjacency_path:     str = "data/preprocessed/adjacency.npy",
    ):
        self.model_dir          = Path(model_dir)
        self.data_path          = Path(data_path)
        self.node_features_path = Path(node_features_path)
        self.adjacency_path     = Path(adjacency_path)

        self.xgb_model   = None
        self.stgcn_model = None
        self.adjacency   = None
        self.adj_tensor  = None
        self.node_features = None     # full array shape (T, N, F)

        self._load_xgboost()
        self._load_stgcn()
        self._load_graph_data()
        self._build_graph()
        self.road_speeds = load_gps_speeds(gps_path)
        self._print_status()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_xgboost(self):
        """Load trained XGBoost model."""
        try:
            import xgboost as xgb
            model_path = self.model_dir / "xgb_traffic.json"
            if model_path.exists():
                self.xgb_model = xgb.XGBRegressor()
                self.xgb_model.load_model(str(model_path))
                print(f"[Router] XGBoost loaded       <- {model_path}")
            else:
                print(f"[Router] XGBoost not found at {model_path}")
        except Exception as e:
            print(f"[Router] XGBoost load failed: {e}")

    def _load_stgcn(self):
        """Load trained STGCN-LSTM model."""
        try:
            import torch
            from models.stgcn_lstm import TrafficSTGCN_LSTM
            model_path = self.model_dir / "stgcn_best.pt"
            adj_path   = self.adjacency_path

            if not model_path.exists():
                print(f"[Router] STGCN checkpoint not found at {model_path}")
                return
            if not adj_path.exists():
                print(f"[Router] Adjacency matrix not found at {adj_path}")
                return

            adj        = np.load(str(adj_path))
            adj_tensor = torch.FloatTensor(adj)
            self.adj_tensor = adj_tensor  # stored for inference

            # Constructor: TrafficSTGCN_LSTM(node_features, gcn_hidden=64,
            #   gcn_out=32, lstm_hidden=128, lstm_layers=2, num_nodes=50, dropout=0.3)
            # Note: adj is stored internally in the model via register_buffer,
            # not passed to the constructor in this architecture.
            model = TrafficSTGCN_LSTM(
                node_features = STGCN_N_FEATURES,  # 6
                num_nodes     = STGCN_N_NODES,      # 10
            )


            checkpoint = torch.load(str(model_path), map_location="cpu")
            # Handle both raw state_dict and wrapped checkpoint
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            self.stgcn_model = model
            print(f"[Router] STGCN-LSTM loaded    <- {model_path}")

        except ImportError as e:
            print(f"[Router] STGCN load skipped (missing dependency: {e})")
        except Exception as e:
            print(f"[Router] STGCN load failed: {e}")

    def _load_graph_data(self):
        """Load pre-built node feature array for STGCN inference."""
        if self.node_features_path.exists():
            self.node_features = np.load(str(self.node_features_path))
            print(f"[Router] Node features loaded <- {self.node_features_path} "
                  f"{self.node_features.shape}")
        else:
            print(f"[Router] Node features not found at {self.node_features_path} "
                  f"— STGCN inference will use CSV fallback.")

    def _build_graph(self):
        self.graph = {i: [] for i in range(len(NODES))}
        for i, j in EDGES:
            self.graph[i].append(j)
            self.graph[j].append(i)

    def _print_status(self):
        xgb_ok   = "OK" if self.xgb_model   else "UNAVAILABLE"
        stgcn_ok = "OK" if self.stgcn_model  else "UNAVAILABLE"
        nf_ok    = "OK" if self.node_features is not None else "UNAVAILABLE"
        print(f"[Router] XGBoost={xgb_ok}  STGCN={stgcn_ok}  NodeFeatures={nf_ok}")
        if self.xgb_model and self.stgcn_model:
            print(f"[Router] Ensemble mode: STGCN×{STGCN_WEIGHT} + XGBoost×{XGB_WEIGHT}")
        elif self.stgcn_model:
            print("[Router] Single-model mode: STGCN only")
        elif self.xgb_model:
            print("[Router] Single-model mode: XGBoost + node residuals")
        else:
            print("[Router] Fallback mode: raw CSV congestion values")

    # ── Timestamp row loader ──────────────────────────────────────────────────

    def _load_row(self, timestamp: Optional[str] = None) -> Tuple[pd.Series, int]:
        """
        Load the CSV row closest to the given timestamp.
        Returns (row, row_index) — row_index is used to slice node_features.
        """
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        if timestamp:
            ts  = _parse_utc(timestamp)
            idx = (df['timestamp'] - ts).abs().idxmin()
        else:
            idx = len(df) - 1

        return df.iloc[idx], int(idx)

    # ── XGBoost per-node scores ───────────────────────────────────────────────

    def _xgboost_node_scores(self, row: pd.Series) -> np.ndarray:
        """
        Derive per-node congestion scores from the XGBoost model.

        XGBoost predicts a single aggregate congestion value for the network.
        We convert this to per-node scores by multiplying by node-specific
        residual corrections (NODE_RESIDUALS) that reflect each intersection's
        typical deviation from the network average — derived from the per-camera
        traffic profiles used in data generation.
        """
        if self.xgb_model is None:
            # Fallback: use raw CSV congestion index
            base = float(row.get('congestion_index', 0.5))
            return np.clip(base * NODE_RESIDUALS, 0.0, 1.0)

        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models.xgboost_model import FEATURE_COLS, build_features

            # Build features on the full CSV so lag/rolling features compute
            # correctly, then take the last row matching our timestamp
            df = pd.read_csv(self.data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = build_features(df)

            # Find the row index matching our loaded row's timestamp
            row_ts = pd.to_datetime(row.get('timestamp'), utc=True)
            if row_ts in df['timestamp'].values:
                target = df[df['timestamp'] == row_ts]
            else:
                target = df.tail(1)

            # Use only the 36 features the model was trained on
            available = [c for c in FEATURE_COLS if c in target.columns]
            xgb_pred  = float(self.xgb_model.predict(target[available].values)[0])
            xgb_pred  = float(np.clip(xgb_pred, 0.0, 1.0))

            # Apply node residuals to produce spatially differentiated scores
            node_scores = np.clip(xgb_pred * NODE_RESIDUALS, 0.0, 1.0)
            return node_scores

        except Exception as e:
            print(f"[Router] XGBoost inference failed ({e}), using CSV fallback.")
            base = float(row.get('congestion_index', 0.5))
            return np.clip(base * NODE_RESIDUALS, 0.0, 1.0)

    # ── STGCN per-node scores ─────────────────────────────────────────────────

    def _stgcn_node_scores(self, row_idx: int) -> Optional[np.ndarray]:
        """
        Derive per-node congestion scores from the STGCN-LSTM model.

        The model takes a (1, T=12, N=10, F=6) input tensor representing
        the last 12 minutes of the road network state and outputs a (10,)
        array of congestion predictions — one per node.

        Returns None if the model or node features are unavailable.
        """
        if self.stgcn_model is None or self.node_features is None:
            return None

        try:
            import torch

            # Build the 12-timestep input window ending at row_idx
            start = max(0, row_idx - STGCN_SEQ_LEN + 1)
            window = self.node_features[start: row_idx + 1]  # shape (<=12, 10, 6)

            # Pad with the first available frame if window is shorter than 12
            if len(window) < STGCN_SEQ_LEN:
                pad = np.tile(window[0:1], (STGCN_SEQ_LEN - len(window), 1, 1))
                window = np.concatenate([pad, window], axis=0)

            # Shape: (1, 12, 10, 6)
            x = torch.FloatTensor(window).unsqueeze(0)

            with torch.no_grad():
                output = self.stgcn_model(x, self.adj_tensor)  # (1,10) or ((1,10), hidden)

            # forward() returns (predictions, hidden_state) tuple
            pred   = output[0] if isinstance(output, tuple) else output
            scores = pred.squeeze(0).numpy()  # shape: (10,)
            return np.clip(scores, 0.0, 1.0)

        except Exception as e:
            print(f"[Router] STGCN inference failed ({e}).")
            return None

    # ── Ensemble congestion scores ────────────────────────────────────────────

    def _get_congestion_scores(
        self,
        timestamp: Optional[str] = None,
    ) -> Dict[int, float]:
        """
        Compute per-node congestion scores using an ensemble of
        STGCN-LSTM and XGBoost predictions.

        Ensemble strategy:
            final[i] = STGCN_WEIGHT * stgcn[i] + XGB_WEIGHT * xgb_node[i]

        If only one model is available, that model's scores are used alone.
        If neither model is available, falls back to raw CSV congestion index
        multiplied by node residuals.
        """
        if not self.data_path.exists():
            return {i: 0.5 for i in range(len(NODES))}

        row, row_idx = self._load_row(timestamp)

        # Get predictions from each model
        xgb_scores   = self._xgboost_node_scores(row)           # shape (10,)
        stgcn_scores = self._stgcn_node_scores(row_idx)         # shape (10,) or None

        # Ensemble
        if stgcn_scores is not None:
            combined = STGCN_WEIGHT * stgcn_scores + XGB_WEIGHT * xgb_scores
            source = "STGCN+XGBoost ensemble"
        else:
            combined = xgb_scores
            source = "XGBoost only"

        combined = np.clip(combined, 0.0, 1.0)
        print(f"[Router] Congestion source: {source}")

        return {i: round(float(combined[i]), 3) for i in range(len(NODES))}

    # ── Incident penalties ────────────────────────────────────────────────────

    def _get_incident_penalties(
        self,
        timestamp: Optional[str] = None,
    ) -> Dict[int, float]:
        """
        Compute per-node incident penalty from ByteTrack incident_count.
        Nodes with active incidents and high congestion receive a penalty
        of up to 0.30 added to their edge weights.
        """
        if not self.data_path.exists():
            return {i: 0.0 for i in range(len(NODES))}

        row, _ = self._load_row(timestamp)

        if 'incident_count' not in row.index:
            return {i: 0.0 for i in range(len(NODES))}

        incident_count = int(row.get('incident_count', 0))
        scores         = self._get_congestion_scores(timestamp)
        penalties      = {}

        for i in range(len(NODES)):
            if incident_count > 0 and scores[i] > 0.7:
                penalties[i] = 0.3 * min(incident_count, 2) / 2
            else:
                penalties[i] = 0.0

        return penalties

    # ── Dijkstra ──────────────────────────────────────────────────────────────

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

    # ── Travel time ───────────────────────────────────────────────────────────

    def _estimate_travel_time(
        self,
        path:       List[int],
        congestion: Dict[int, float],
    ) -> float:
        total_time = 0.0
        for idx in range(len(path) - 1):
            u, v            = path[idx], path[idx+1]
            dist_km         = EDGE_DISTANCES.get((u, v), 1.0)
            free_flow_speed = self.road_speeds.get((u, v),
                              self.road_speeds.get((v, u), 30.0))
            avg_cong        = (congestion[u] + congestion[v]) / 2
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

    # ── Public API ────────────────────────────────────────────────────────────

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
             overall_congestion, overall_congestion_label, all_node_congestion,
             congestion_source  (which models were used)
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

        # Record which models contributed to this prediction
        if self.stgcn_model and self.xgb_model:
            source = f"STGCN-LSTM x{STGCN_WEIGHT} + XGBoost x{XGB_WEIGHT} ensemble"
        elif self.stgcn_model:
            source = "STGCN-LSTM only"
        elif self.xgb_model:
            source = "XGBoost + node residuals"
        else:
            source = "CSV fallback"

        return {
            "origin":                   origin,
            "destination":              destination,
            "timestamp":                timestamp or "latest",
            "congestion_source":        source,
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
        print("\n" + "=" * 60)
        print("  KOLKATA TRAFFIC ROUTE SUGGESTION")
        print("=" * 60)
        print(f"  From   : {result['origin']}")
        print(f"  To     : {result['destination']}")
        print(f"  Time   : {result['timestamp']}")
        print(f"  Source : {result['congestion_source']}")
        print("-" * 60)
        print(f"  Recommended Route ({len(result['path'])} stops):")
        for i, node in enumerate(result['path']):
            marker = "  [START]" if i == 0 else \
                     ("  [ END ]" if i == len(result['path'])-1 else f"  [  {i:2d}  ]")
            cong   = result['all_node_congestion'].get(node, 0)
            print(f"{marker} {node:30s} Congestion: {cong:.2f} ({self._congestion_label(cong)})")
        print("-" * 60)
        for i, (seg_cong, seg_dist, color, spd) in enumerate(zip(
            result['congestion_per_segment'],
            result['segment_distances_km'],
            result['segment_colors'],
            result['segment_speeds_kmh'],
        )):
            print(f"  Segment {i+1}: {result['path'][i]} -> {result['path'][i+1]}")
            print(f"           Distance: {seg_dist} km | Free-flow: {spd} km/h | "
                  f"Congestion: {seg_cong:.2f} | Status: {color.upper()}")
        print("-" * 60)
        print(f"  Total Distance    : {result['total_distance_km']} km")
        print(f"  Est. Travel Time  : {result['travel_time_mins']} mins")
        print(f"  Overall Congestion: {result['overall_congestion']:.2f} "
              f"({result['overall_congestion_label']})")
        print("=" * 60)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def list_locations():
    print("\nAvailable locations:")
    for i, meta in NODES.items():
        print(f"  {i:2d}. {meta['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kolkata Traffic Route Suggester — STGCN-LSTM + XGBoost ensemble"
    )
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