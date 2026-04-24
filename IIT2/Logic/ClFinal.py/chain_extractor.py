"""
chain_extractor.py  (v2 — GA zone partitioning + per-zone DBSCAN)
------------------------------------------------------------------
Pipeline:
  1. Load stay-points CSV  (device_aid, lat, lon, arrival_time)
  2. Extract first-mile and last-mile demand points per station
     (kept exactly from v1 — logic unchanged)
  3. For each station's demand points:
       a. GA partitions points into N zones (N tried 2..N_MAX, best wins)
          Fitness = α·coverage  −  β·avg_road_distance  −  γ·imbalance
       b. DBSCAN runs inside each GA zone → virtual stop centroids
       c. Centroids are map-matched to primary/secondary roads via road_matcher
  4. Save:
       - first_mile_feeder_demand.csv       (raw demand points, unchanged)
       - last_mile_dispersal_demand.csv     (raw demand points, unchanged)
       - virtual_stops_<station>.csv        (road-snapped stops per station)
       - zone_summary.csv                   (N chosen, fitness, coverage per station)

Dependencies: road_matcher.py (must be in same folder)
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# Our road utility module
import road_matcher as rm

# ── GA configuration ──────────────────────────────────────────────────────────

GA_CONFIG = {
    "n_min":          2,      # minimum routes (zones) per station
    "n_max":          6,      # maximum routes (zones) per station
    "population_size": 90,    # number of chromosomes per generation
    "generations":    120,    # how many generations to evolve
    "mutation_rate":  0.12,   # probability of flipping a point's zone
    "elite_frac":     0.15,   # top fraction kept unchanged each generation
    # Fitness weights — adjust to shift priority
    "alpha":          0.50,   # weight for coverage (higher = prioritise coverage)
    "beta":           0.30,   # weight for avg road distance (higher = tighter zones)
    "gamma":          0.20,   # weight for zone balance (higher = more even zones)
    # A demand point is "covered" if it's within this metres of its zone centroid
    "coverage_radius_m": 400,
}

# ── DBSCAN configuration (applied per zone) ───────────────────────────────────

DBSCAN_CONFIG = {
    # epsilon in metres — converted to radians internally
    # Smaller than global DBSCAN because zones are already tight
    "eps_m":      250,
    "min_samples": 3,
}

# ── Trip extraction (unchanged from v1) ───────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    """Fast haversine in metres — used only for trip filtering, not metrics."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _tag_station(lat, lon, stations_dict, radius=600):
    for name, coords in stations_dict.items():
        if _haversine(lat, lon, coords[0], coords[1]) <= radius:
            return name, coords
    return "Non-Station", None


def extract_trip_chains(df, stations_dict, max_bus_radius=6000):
    """
    Extract first-mile and last-mile demand points from stay-points DataFrame.
    Logic identical to v1 — no changes here.

    Args:
        df             : Stay-points DataFrame (device_aid, lat, lon, arrival_time)
        stations_dict  : {name: (lat, lon)}
        max_bus_radius : Max feeder distance in metres (default 6km)

    Returns:
        first_mile_df, last_mile_df  — DataFrames of demand points
    """
    lat_col  = next(c for c in ["lat", "latitude",  "stop_lat"]  if c in df.columns)
    lon_col  = next(c for c in ["lon", "longitude", "stop_lon"]  if c in df.columns)
    time_col = "arrival_time" if "arrival_time" in df.columns else "timestamp"

    df = df.copy()
    df[time_col]    = pd.to_datetime(df[time_col])
    df["trip_date"] = df[time_col].dt.date
    df = df.sort_values(["device_aid", time_col]).reset_index(drop=True)

    print("  Geofencing station catchments (600m radius)...")
    tags = df.apply(
        lambda row: _tag_station(row[lat_col], row[lon_col], stations_dict), axis=1
    )
    df["location_tag"]    = [t[0] for t in tags]
    df["station_coords"]  = [t[1] for t in tags]

    first_mile, last_mile = [], []

    for (device_id, trip_date), day in df.groupby(["device_aid", "trip_date"]):
        day = day.reset_index(drop=True)
        station_idx = day[day["location_tag"] != "Non-Station"].index

        for idx in station_idx:
            sname        = day.loc[idx, "location_tag"]
            s_lat, s_lon = day.loc[idx, "station_coords"]

            # First mile: last non-station point BEFORE reaching station
            past    = day.loc[:idx-1]
            origins = past[past["location_tag"] == "Non-Station"]
            if not origins.empty:
                home = origins.iloc[-1]
                d = _haversine(home[lat_col], home[lon_col], s_lat, s_lon)
                if 200 < d <= max_bus_radius:
                    first_mile.append({
                        "device_aid": device_id, "date": trip_date,
                        "station": sname, "type": "Feeder_Demand",
                        "lat": home[lat_col], "lon": home[lon_col],
                    })

            # Last mile: first non-station point AFTER leaving station
            future = day.loc[idx+1:]
            dests  = future[future["location_tag"] == "Non-Station"]
            if not dests.empty:
                office = dests.iloc[0]
                d = _haversine(office[lat_col], office[lon_col], s_lat, s_lon)
                if 200 < d <= max_bus_radius:
                    last_mile.append({
                        "device_aid": device_id, "date": trip_date,
                        "station": sname, "type": "Dispersal_Demand",
                        "lat": office[lat_col], "lon": office[lon_col],
                    })

    return pd.DataFrame(first_mile), pd.DataFrame(last_mile)


# ── GA zone partitioning ───────────────────────────────────────────────────────

def _zone_centroids(points, assignments, n_zones):
    """Return (lat, lon) centroid for each zone. zones with 0 points get station centre."""
    centroids = []
    for z in range(n_zones):
        mask = assignments == z
        if mask.sum() == 0:
            centroids.append((np.mean(points[:, 0]), np.mean(points[:, 1])))
        else:
            centroids.append((points[mask, 0].mean(), points[mask, 1].mean()))
    return centroids


def _fitness(assignments, points, road_dist_matrix, n_zones, cfg):
    """
    Score a chromosome (zone assignment array).

    Higher is better.

    Components:
      coverage   : fraction of points within coverage_radius_m of their zone centroid
      avg_dist   : mean road distance from each point to its zone centroid (normalised)
      imbalance  : std of zone sizes / mean zone size (0 = perfectly balanced)
    """
    n          = len(points)
    centroids  = _zone_centroids(points, assignments, n_zones)
    coverage_radius = cfg["coverage_radius_m"]

    covered   = 0
    total_dist = 0.0
    zone_sizes = np.zeros(n_zones)

    for i, z in enumerate(assignments):
        c_lat, c_lon = centroids[z]
        # Use road distance if available, haversine as fallback for speed
        # (full road matrix pre-computed once per station outside GA loop)
        zone_sizes[z] += 1
        # Find centroid index in points array for matrix lookup
        # We use haversine here for intra-zone centroid distance to keep GA fast;
        # road distances are used in the final metric computation.
        d = rm.haversine_distance(points[i, 0], points[i, 1], c_lat, c_lon)
        total_dist += d
        if d <= coverage_radius:
            covered += 1

    coverage   = covered / n
    avg_dist   = total_dist / n                          # metres
    norm_dist  = avg_dist / 6000.0                       # normalise to ~[0,1] for 6km max
    imbalance  = (zone_sizes.std() / zone_sizes.mean()
                  if zone_sizes.mean() > 0 else 1.0)

    return (cfg["alpha"] * coverage
            - cfg["beta"] * norm_dist
            - cfg["gamma"] * imbalance)


def _run_ga(points, n_zones, road_dist_matrix, cfg, station_name):
    """
    Run GA for a fixed n_zones. Returns (best_assignments, best_fitness).

    Chromosome: integer array of length n_points, values in [0, n_zones).
    Each value is the zone ID assigned to that demand point.
    """
    n          = len(points)
    pop_size   = cfg["population_size"]
    n_elite    = max(1, int(pop_size * cfg["elite_frac"]))
    mut_rate   = cfg["mutation_rate"]

    # Initialise population randomly
    population = [np.random.randint(0, n_zones, size=n) for _ in range(pop_size)]

    best_fitness  = -np.inf
    best_solution = population[0].copy()

    for gen in range(cfg["generations"]):
        # Score all chromosomes
        scores = [_fitness(chrom, points, road_dist_matrix, n_zones, cfg)
                  for chrom in population]

        # Track best
        gen_best_idx = int(np.argmax(scores))
        if scores[gen_best_idx] > best_fitness:
            best_fitness  = scores[gen_best_idx]
            best_solution = population[gen_best_idx].copy()

        # Sort by fitness descending
        ranked = sorted(zip(scores, range(pop_size)), reverse=True)
        elite  = [population[ranked[i][1]].copy() for i in range(n_elite)]

        # Build next generation
        next_pop = elite[:]
        while len(next_pop) < pop_size:
            # Tournament selection (k=3)
            p1 = population[random.choice(ranked[:20])[1]].copy()
            p2 = population[random.choice(ranked[:20])[1]].copy()

            # Single-point crossover
            cut   = random.randint(1, n - 1)
            child = np.concatenate([p1[:cut], p2[cut:]])

            # Mutation
            mask          = np.random.random(n) < mut_rate
            child[mask]   = np.random.randint(0, n_zones, size=mask.sum())

            next_pop.append(child)

        population = next_pop

    return best_solution, best_fitness


def ga_zone_partition(demand_points_df, station_name, station_graph, cfg=None):
    """
    Partition demand points into optimal zones using a Genetic Algorithm.

    Tries N = n_min .. n_max zones, picks the N with best fitness score.
    This means each station gets its own natural number of routes.

    Args:
        demand_points_df : DataFrame with 'lat' and 'lon' columns
        station_name     : String, used for logging
        station_graph    : osmnx graph from road_matcher (used for snapping later)
        cfg              : GA config dict (defaults to GA_CONFIG)

    Returns:
        demand_points_df with added column 'zone' (int, 0-indexed)
        best_n           : int — optimal number of zones chosen
        fitness_scores   : dict {n: score} for all tested N values
    """
    if cfg is None:
        cfg = GA_CONFIG

    points = demand_points_df[["lat", "lon"]].values
    n      = len(points)

    if n < cfg["n_min"] * 2:
        # Too few points — put everything in zone 0
        print(f"  [GA] {station_name}: only {n} points, using 1 zone.")
        out = demand_points_df.copy()
        out["zone"] = 0
        return out, 1, {1: 1.0}

    print(f"  [GA] {station_name}: {n} points, testing N = "
          f"{cfg['n_min']}..{cfg['n_max']} zones ...")

    # Pre-compute road distance matrix once (reused across all N trials)
    # For GA speed we skip this and use haversine inside _fitness,
    # only using the road matrix for final stop evaluation.
    road_dist_matrix = None   # placeholder — road matrix used in eval2, not GA loop

    best_n        = cfg["n_min"]
    best_fitness  = -np.inf
    best_assign   = None
    fitness_scores = {}

    for n_zones in range(cfg["n_min"], cfg["n_max"] + 1):
        assign, score = _run_ga(points, n_zones, road_dist_matrix, cfg, station_name)
        fitness_scores[n_zones] = round(score, 4)
        print(f"      N={n_zones}: fitness = {score:.4f}")

        if score > best_fitness:
            best_fitness = score
            best_n       = n_zones
            best_assign  = assign.copy()

    print(f"  [GA] {station_name}: optimal N = {best_n}  "
          f"(fitness = {best_fitness:.4f})")

    out = demand_points_df.copy()
    out["zone"] = best_assign
    return out, best_n, fitness_scores


# ── Per-zone DBSCAN → virtual stops ───────────────────────────────────────────

def dbscan_per_zone(zoned_df, station_name, station_graph, dbscan_cfg=None):
    """
    Run DBSCAN inside each GA zone to find virtual bus stop locations,
    then snap each centroid to the road network.

    Args:
        zoned_df       : DataFrame with 'lat', 'lon', 'zone' columns
        station_name   : String, for logging
        station_graph  : osmnx graph for this station
        dbscan_cfg     : DBSCAN config dict (defaults to DBSCAN_CONFIG)

    Returns:
        virtual_stops_df : DataFrame with one row per stop:
            zone, stop_id, lat, lon,          (centroid before snapping)
            snapped_lat, snapped_lon,          (after road snap)
            unique_commuters, snap_distance,
            serviceable, station
    """
    if dbscan_cfg is None:
        dbscan_cfg = DBSCAN_CONFIG

    eps_rad = dbscan_cfg["eps_m"] / 6_371_000.0   # metres → radians for haversine DBSCAN

    all_stops = []
    stop_counter = 0

    for zone_id in sorted(zoned_df["zone"].unique()):
        zone_pts = zoned_df[zoned_df["zone"] == zone_id].reset_index(drop=True)

        if len(zone_pts) < dbscan_cfg["min_samples"]:
            # Zone too small for DBSCAN — use centroid directly
            centroid_lat = zone_pts["lat"].mean()
            centroid_lon = zone_pts["lon"].mean()
            all_stops.append({
                "zone":             zone_id,
                "stop_id":          stop_counter,
                "lat":              centroid_lat,
                "lon":              centroid_lon,
                "unique_commuters": len(zone_pts),
                "station":          station_name,
            })
            stop_counter += 1
            continue

        coords = np.radians(zone_pts[["lat", "lon"]].values)
        labels = DBSCAN(
            eps=eps_rad,
            min_samples=dbscan_cfg["min_samples"],
            algorithm="ball_tree",
            metric="haversine",
        ).fit_predict(coords)

        zone_pts = zone_pts.copy()
        zone_pts["cluster"] = labels

        # Each DBSCAN cluster = one virtual stop (centroid of cluster)
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue   # noise points — not assigned to any stop
            cluster_pts = zone_pts[zone_pts["cluster"] == cluster_id]
            all_stops.append({
                "zone":             zone_id,
                "stop_id":          stop_counter,
                "lat":              cluster_pts["lat"].mean(),
                "lon":              cluster_pts["lon"].mean(),
                "unique_commuters": len(cluster_pts),
                "station":          station_name,
            })
            stop_counter += 1

    if not all_stops:
        print(f"  [DBSCAN] {station_name}: no clusters found — check data density.")
        return pd.DataFrame()

    stops_df = pd.DataFrame(all_stops)

    # Map-match all centroids to primary/secondary roads
    print(f"  [DBSCAN] {station_name}: {len(stops_df)} virtual stops found. "
          f"Snapping to road ...")
    stops_df = rm.snap_stops_batch(stops_df, station_graph)
    stops_df["station"] = station_name

    return stops_df


# ── Main pipeline function ────────────────────────────────────────────────────

def run_extraction_pipeline(
    stops_file_path,
    stations_dict,
    output_dir,
    max_bus_radius=6000,
    ga_cfg=None,
    dbscan_cfg=None,
    force_graph_download=False,
):
    """
    Full pipeline: stay-points CSV → demand points → GA zones →
                   DBSCAN stops → road-snapped virtual stops.

    Args:
        stops_file_path      : Path to stay-points CSV
        stations_dict        : {name: (lat, lon)}
        output_dir           : Where to save all outputs
        max_bus_radius       : Max feeder distance metres (default 6km)
        ga_cfg               : Override GA_CONFIG defaults
        dbscan_cfg           : Override DBSCAN_CONFIG defaults
        force_graph_download : Re-download OSM graphs even if cached

    Outputs saved to output_dir:
        first_mile_feeder_demand.csv
        last_mile_dispersal_demand.csv
        virtual_stops_<station>.csv      (one per station)
        zone_summary.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"CHAIN EXTRACTOR v2")
    print(f"{'='*55}")
    print(f"Loading: {stops_file_path}")
    df = pd.read_csv(stops_file_path)
    print(f"Loaded {len(df):,} stay-points.")

    # ── Step 2: Load all station road graphs ─────────────────────────────────
    print(f"\nLoading road graphs for {len(stations_dict)} stations ...")
    station_graphs = rm.preload_all_stations(
        stations_dict, force_download=force_graph_download
    )

    # ── Step 3: Extract trip chains ──────────────────────────────────────────
    print(f"\nExtracting first-mile and last-mile demand points ...")
    first_mile_df, last_mile_df = extract_trip_chains(
        df, stations_dict, max_bus_radius
    )
    print(f"  First-mile demand points : {len(first_mile_df):,}")
    print(f"  Last-mile demand points  : {len(last_mile_df):,}")

    # Save raw demand CSVs (unchanged format from v1)
    first_mile_df.to_csv(
        os.path.join(output_dir, "first_mile_feeder_demand.csv"), index=False
    )
    last_mile_df.to_csv(
        os.path.join(output_dir, "last_mile_dispersal_demand.csv"), index=False
    )

    # ── Step 4: GA + DBSCAN per station ─────────────────────────────────────
    all_virtual_stops = []
    zone_summary_rows = []

    for station_name, (s_lat, s_lon) in stations_dict.items():
        print(f"\n{'─'*50}")
        print(f"Processing: {station_name}")
        print(f"{'─'*50}")

        G = station_graphs[station_name]

        # Combine first and last mile for this station
        fm = first_mile_df[first_mile_df["station"] == station_name].copy()
        lm = last_mile_df[last_mile_df["station"]  == station_name].copy()
        demand = pd.concat([fm, lm], ignore_index=True)

        if demand.empty:
            print(f"  [SKIP] No demand points found for {station_name}.")
            continue

        print(f"  Demand points: {len(demand):,} "
              f"({len(fm):,} first-mile, {len(lm):,} last-mile)")

        # GA zone partitioning
        zoned_demand, best_n, fitness_scores = ga_zone_partition(
            demand, station_name, G, cfg=ga_cfg
        )

        # Per-zone DBSCAN → road-snapped virtual stops
        virtual_stops = dbscan_per_zone(
            zoned_demand, station_name, G, dbscan_cfg=dbscan_cfg
        )

        if virtual_stops.empty:
            continue

        # Save per-station stops file
        out_path = os.path.join(output_dir, f"virtual_stops_{station_name}.csv")
        virtual_stops.to_csv(out_path, index=False)
        print(f"  Saved {len(virtual_stops)} stops → {out_path}")

        all_virtual_stops.append(virtual_stops)

        # Zone summary row
        serviceable_pct = (virtual_stops["serviceable"].sum()
                           / len(virtual_stops) * 100
                           if len(virtual_stops) > 0 else 0)
        zone_summary_rows.append({
            "station":           station_name,
            "optimal_n_zones":   best_n,
            "n_virtual_stops":   len(virtual_stops),
            "total_demand_pts":  len(demand),
            "serviceable_pct":   round(serviceable_pct, 1),
            "best_ga_fitness":   round(fitness_scores.get(best_n, 0), 4),
            "fitness_by_n":      str(fitness_scores),
        })

    # ── Step 5: Save summary ─────────────────────────────────────────────────
    if zone_summary_rows:
        summary_df = pd.DataFrame(zone_summary_rows)
        summary_path = os.path.join(output_dir, "zone_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*55}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*55}")
        print(summary_df[[
            "station", "optimal_n_zones", "n_virtual_stops",
            "total_demand_pts", "serviceable_pct", "best_ga_fitness"
        ]].to_string(index=False))
        print(f"\nZone summary saved → {summary_path}")

    return all_virtual_stops, zone_summary_rows


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    STOPS_DATA_FILE = "mumbai_multiday_stops_robust.csv"   # ← update if needed
    OUTPUT_DIRECTORY = "./demand_matrices_v2/"

    STATIONS = {
        "Andheri":    (19.1197, 72.8464),
        "Bandra":     (19.0544, 72.8402),
        "Borivali":   (19.2291, 72.8573),
        "Goregaon":   (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264),
    }

    run_extraction_pipeline(
        stops_file_path=STOPS_DATA_FILE,
        stations_dict=STATIONS,
        output_dir=OUTPUT_DIRECTORY,
        max_bus_radius=6000,
    )