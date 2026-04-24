"""
road_matcher.py  (v2 — per-station graphs)
-------------------------------------------
Downloads a small road graph around each station instead of the full
Mumbai bounding box. This avoids OSM Overpass timeouts and is fast.

Each station graph is cached as  road_cache/<station_name>_graph.graphml
and loads in ~2s on subsequent runs.

Only PRIMARY and SECONDARY roads are included — buses won't enter
smaller lanes.

Public API
----------
get_station_graph(station_name, lat, lon)        → nx.MultiDiGraph
preload_all_stations(stations)                   → dict of graphs
snap_to_road(lat, lon, G)                        → dict
snap_stops_batch(stops_df, G)                    → DataFrame
get_road_distance(lat1,lon1, lat2,lon2, G)       → float (metres)
get_travel_time_minutes(lat1,lon1, lat2,lon2, G) → float (minutes)
get_road_distance_matrix(points, G)              → np.ndarray
get_travel_time_matrix(points, G)                → np.ndarray
haversine_distance(lat1,lon1, lat2,lon2)         → float (metres)
"""

import os
import time
import warnings
import numpy as np
import networkx as nx

try:
    import osmnx as ox
except ImportError:
    raise ImportError("osmnx is required.  Run:  pip install osmnx")

# ── Configuration ─────────────────────────────────────────────────────────────

# Radius around each station to download (metres).
# 7 km covers typical feeder catchment + buffer.
STATION_RADIUS_M = 7000

# OSM road filter — primary and secondary only
ROAD_FILTER = (
    '["highway"~"primary|secondary|primary_link|secondary_link"]'
    '["access"!~"private"]'
)

# Cache folder (created next to this script automatically)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "road_cache")

# Stop centroids further than this from any eligible road are UNSERVICEABLE
MAX_SNAP_DISTANCE_M = 500.0

# Fallback bus speed used only when edge travel_time attribute is missing
AVG_BUS_SPEED_KMH = 20.0

# OSM Overpass timeout (seconds) — raise if you have a slow connection
DOWNLOAD_TIMEOUT = 180


# ── Graph loading (per station) ───────────────────────────────────────────────

def get_station_graph(
    station_name: str,
    lat: float,
    lon: float,
    force_download: bool = False,
) -> nx.MultiDiGraph:
    """
    Load (or download) the road graph for one station.

    Downloads a circle of STATION_RADIUS_M around (lat, lon),
    keeping only primary + secondary roads, then caches to disk.

    Args:
        station_name  : Human-readable name, used for cache filename.
        lat, lon      : Station centre coordinates.
        force_download: If True, re-download even if a cache file exists.

    Returns:
        nx.MultiDiGraph with 'length' and 'travel_time' edge attributes.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name  = station_name.replace(" ", "_").lower()
    cache_path = os.path.join(CACHE_DIR, f"{safe_name}_graph.graphml")

    if not force_download and os.path.exists(cache_path):
        print(f"  [road_matcher] Loading cached graph: {station_name}")
        G = ox.load_graphml(cache_path)
        print(
            f"  [road_matcher] {station_name}: "
            f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges"
        )
        return G

    print(
        f"  [road_matcher] Downloading graph for {station_name} "
        f"(radius = {STATION_RADIUS_M/1000:.0f} km) ..."
    )
    # osmnx 2.x uses requests_timeout instead of timeout
    ox.settings.requests_timeout = DOWNLOAD_TIMEOUT

    t0 = time.time()
    try:
        G = ox.graph_from_point(
            center_point=(lat, lon),
            dist=STATION_RADIUS_M,
            network_type="drive",
            custom_filter=ROAD_FILTER,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Download failed for {station_name}: {exc}\n"
            "Check your internet connection and try again."
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

    ox.save_graphml(G, cache_path)
    print(
        f"  [road_matcher] {station_name}: downloaded in {time.time()-t0:.1f}s  "
        f"({G.number_of_nodes():,} nodes)  → cached at {cache_path}"
    )
    return G


def preload_all_stations(
    stations: dict,
    force_download: bool = False,
) -> dict:
    """
    Download / load graphs for every station upfront.

    Args:
        stations: {name: (lat, lon)}  — same dict used in chain_extractor.py
        force_download: re-download all even if caches exist

    Returns:
        {name: nx.MultiDiGraph}
    """
    print(f"\n[road_matcher] Loading graphs for {len(stations)} station(s) ...")
    graphs = {}
    for name, (lat, lon) in stations.items():
        graphs[name] = get_station_graph(name, lat, lon,
                                         force_download=force_download)
    print("[road_matcher] All station graphs ready.\n")
    return graphs


# ── Snapping ──────────────────────────────────────────────────────────────────

def snap_to_road(lat: float, lon: float, G: nx.MultiDiGraph) -> dict:
    """
    Snap a lat/lon point to the nearest node on an eligible road in G.

    Returns dict with keys:
        snapped_lat    float  — latitude of nearest road node
        snapped_lon    float  — longitude of nearest road node
        node_id        int    — OSM node ID
        snap_distance  float  — straight-line metres from input to snapped node
        serviceable    bool   — False if snap_distance > MAX_SNAP_DISTANCE_M
    """
    node_id, snap_dist = ox.distance.nearest_nodes(G, X=lon, Y=lat,
                                                    return_dist=True)
    nd = G.nodes[node_id]
    return {
        "snapped_lat":   nd["y"],
        "snapped_lon":   nd["x"],
        "node_id":       node_id,
        "snap_distance": float(snap_dist),
        "serviceable":   float(snap_dist) <= MAX_SNAP_DISTANCE_M,
    }


def snap_stops_batch(stops_df, G: nx.MultiDiGraph):
    """
    Snap a DataFrame of virtual stops to the road network.

    Args:
        stops_df : DataFrame with at minimum columns 'lat' and 'lon'.
        G        : Station graph from get_station_graph().

    Returns:
        Copy of stops_df with added columns:
        snapped_lat, snapped_lon, node_id, snap_distance, serviceable.
    """
    results = [snap_to_road(row.lat, row.lon, G)
               for row in stops_df.itertuples(index=False)]

    out = stops_df.copy()
    out["snapped_lat"]   = [r["snapped_lat"]   for r in results]
    out["snapped_lon"]   = [r["snapped_lon"]   for r in results]
    out["node_id"]       = [r["node_id"]       for r in results]
    out["snap_distance"] = [r["snap_distance"] for r in results]
    out["serviceable"]   = [r["serviceable"]   for r in results]

    n_total = len(out)
    n_ok    = int(out["serviceable"].sum())
    n_bad   = n_total - n_ok
    if n_bad:
        print(
            f"  [road_matcher] Snapped {n_ok}/{n_total} stops  "
            f"({n_bad} UNSERVICEABLE — >{MAX_SNAP_DISTANCE_M:.0f}m from road)"
        )
    else:
        print(f"  [road_matcher] All {n_total} stops snapped successfully.")
    return out


# ── Distance & travel time ────────────────────────────────────────────────────

def _nearest_nodes(lat1, lon1, lat2, lon2, G):
    n1 = ox.distance.nearest_nodes(G, X=lon1, Y=lat1)
    n2 = ox.distance.nearest_nodes(G, X=lon2, Y=lat2)
    return n1, n2


def get_road_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    G: nx.MultiDiGraph,
) -> float:
    """Shortest road-network distance between two points (metres). np.inf if unreachable."""
    n1, n2 = _nearest_nodes(lat1, lon1, lat2, lon2, G)
    try:
        return float(nx.shortest_path_length(G, n1, n2, weight="length"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return np.inf


def get_travel_time_minutes(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    G: nx.MultiDiGraph,
) -> float:
    """Estimated travel time between two points (minutes). np.inf if unreachable."""
    n1, n2 = _nearest_nodes(lat1, lon1, lat2, lon2, G)
    try:
        secs = nx.shortest_path_length(G, n1, n2, weight="travel_time")
        return float(secs) / 60.0
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return np.inf


def get_road_distance_matrix(points: list, G: nx.MultiDiGraph) -> np.ndarray:
    """
    N×N road-distance matrix for a list of (lat, lon) tuples (metres).
    Diagonal = 0.  Unreachable pairs = np.inf.
    Called by the GA fitness function and the TSP sequencer.
    """
    n     = len(points)
    nodes = [ox.distance.nearest_nodes(G, X=lon, Y=lat) for lat, lon in points]
    mat   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                mat[i, j] = nx.shortest_path_length(
                    G, nodes[i], nodes[j], weight="length"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                mat[i, j] = np.inf
    return mat


def get_travel_time_matrix(points: list, G: nx.MultiDiGraph) -> np.ndarray:
    """
    N×N travel-time matrix (minutes) for a list of (lat, lon) tuples.
    Same structure as get_road_distance_matrix.
    """
    n     = len(points)
    nodes = [ox.distance.nearest_nodes(G, X=lon, Y=lat) for lat, lon in points]
    mat   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                secs = nx.shortest_path_length(
                    G, nodes[i], nodes[j], weight="travel_time"
                )
                mat[i, j] = secs / 60.0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                mat[i, j] = np.inf
    return mat


# ── Haversine (straight-line fallback) ───────────────────────────────────────

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Straight-line Haversine distance (metres).
    Use only as a fast pre-filter — never for actual route metrics.
    """
    R       = 6_371_000
    phi1    = np.radians(lat1)
    phi2    = np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    print("=" * 55)
    print("road_matcher.py v2 — smoke test")
    print(f"osmnx version : {ox.__version__}")
    print(f"Cache folder  : {CACHE_DIR}")
    print("=" * 55)

    # Only Andheri for the quick test — downloads in ~10s
    STATION_NAME = "Andheri"
    STATION_LAT  = 19.1197
    STATION_LON  = 72.8464

    print(f"\n[1] Load graph for {STATION_NAME} ...")
    G = get_station_graph(STATION_NAME, STATION_LAT, STATION_LON)

    print(f"\n[2] Snap station point to road ...")
    snap = snap_to_road(STATION_LAT, STATION_LON, G)
    print(f"    Original : ({STATION_LAT}, {STATION_LON})")
    print(f"    Snapped  : ({snap['snapped_lat']:.5f}, {snap['snapped_lon']:.5f})")
    print(f"    Snap dist: {snap['snap_distance']:.1f} m  |  "
          f"Serviceable: {snap['serviceable']}")

    # Two nearby points
    P1 = (19.1197, 72.8464)
    P2 = (19.1050, 72.8350)

    print(f"\n[3] Road distance P1 → P2 ...")
    d = get_road_distance(*P1, *P2, G)
    hs = haversine_distance(*P1, *P2)
    print(f"    Road  : {d/1000:.2f} km")
    print(f"    Straight-line: {hs/1000:.2f} km  "
          f"(circuity = {d/hs:.2f}x)" if hs > 0 else "")

    print(f"\n[4] Travel time P1 → P2 ...")
    tt = get_travel_time_minutes(*P1, *P2, G)
    print(f"    {tt:.1f} minutes")

    print(f"\n[5] 3×3 distance matrix ...")
    P3  = (19.1300, 72.8550)
    mat = get_road_distance_matrix([P1, P2, P3], G)
    for i, row in enumerate(mat):
        print(f"    P{i+1}: " + "  ".join(f"{v/1000:5.2f}km" for v in row))

    print(f"\n[6] Batch snap ...")
    df = pd.DataFrame({
        "lat":   [P1[0], P2[0], P3[0]],
        "lon":   [P1[1], P2[1], P3[1]],
        "label": ["Station", "NearbyA", "NearbyB"],
    })
    snapped_df = snap_stops_batch(df, G)
    cols = ["label", "snapped_lat", "snapped_lon", "snap_distance", "serviceable"]
    print(snapped_df[cols].to_string(index=False))

    print("\n" + "=" * 55)
    print("Smoke test complete — road_matcher is working.")
    print("=" * 55)