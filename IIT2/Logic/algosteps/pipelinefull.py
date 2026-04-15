import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
import random
import copy
import os

# ==========================================
# HELPER FUNCTIONS (MATH & DISTANCE)
# ==========================================
def get_distance(lat_array, lon_array, target_lat, target_lon):
    """Calculates Haversine distance in meters to a target coordinate."""
    R = 6371000  
    phi1, phi2 = np.radians(lat_array), np.radians(target_lat)
    dphi, dlambda = np.radians(target_lat - lat_array), np.radians(target_lon - lon_array)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def build_distance_matrix(station_lat, station_lon, stops):
    """Precomputes all distances for the Genetic Algorithm."""
    all_points = [{'lat': station_lat, 'lon': station_lon}] + stops
    n = len(all_points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = get_distance(all_points[i]['lat'], all_points[i]['lon'], 
                                            all_points[j]['lat'], all_points[j]['lon'])
    return matrix

def compute_dwell_times(df, time_col='timestamp'):
    df = df.sort_values(by=['device_aid', time_col])
    
    # Convert to datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Calculate time difference between consecutive points
    df['prev_time'] = df.groupby('device_aid')[time_col].shift(1)
    df['dwell_time_sec'] = (df[time_col] - df['prev_time']).dt.total_seconds()
    
    return df
# ==========================================
# PHASE 1: CATCHMENT EXTRACTION
# ==========================================
def phase1_extract_catchment(stops_file, station_name, min_lat, max_lat, min_lon, max_lon):
    print("\n--- PHASE 1: CATCHMENT EXTRACTION ---")
    print(f"Loading master stops data from {stops_file}...")
    try:
        df = pd.read_csv(stops_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{stops_file}'.")
        return None

    # Ensure dwell_time_sec exists regardless of source schema.
    if 'dwell_time_sec' not in df.columns:
        if 'duration_mins' in df.columns:
            df['dwell_time_sec'] = pd.to_numeric(df['duration_mins'], errors='coerce') * 60
        elif {'arrival_time', 'departure_time'}.issubset(df.columns):
            arrival = pd.to_datetime(df['arrival_time'], errors='coerce')
            departure = pd.to_datetime(df['departure_time'], errors='coerce')
            df['dwell_time_sec'] = (departure - arrival).dt.total_seconds()
        else:
            print("Error: Could not derive 'dwell_time_sec' from input data.")
            return None

    # Find transit stops strictly inside the origin rectangle
    transit_stops = df[
        (df['stop_lat'] >= min_lat) & (df['stop_lat'] <= max_lat) &
        (df['stop_lon'] >= min_lon) & (df['stop_lon'] <= max_lon) &
        (df['likely_purpose'] == 'Other / Transit') &
        (df['dwell_time_sec'] >= 300)
    ]
    
    target_commuters = transit_stops['device_aid'].unique()
    print(f"Found {len(target_commuters):,} true train commuters using the Rectangular Geofence!")

    if len(target_commuters) == 0:
        return None

    commuter_df = df[df['device_aid'].isin(target_commuters)].copy()
    # Save checkpoint
    commuter_df.to_csv(f"{station_name.lower().replace(' ', '_')}_commuters_checkpoint.csv", index=False)
    return commuter_df

# ==========================================
# PHASE 2: AI VIRTUAL STOPS (DBSCAN)
# ==========================================
def phase2_generate_virtual_stops(commuter_df, station_lat, station_lon):
    print("\n--- PHASE 2: AI VIRTUAL BUS STOPS ---")
    destinations = commuter_df[commuter_df['likely_purpose'] != 'Other / Transit'].copy()
    print(f"Total initial destinations: {len(destinations):,}")

    # Exclusion Zones (Competing Stations)
    competing_stations = {
        "Jogeshwari Station": (19.1363, 72.8489),
        "Vile Parle Station": (19.1006, 72.8440),
        "Goregaon Station": (19.1645, 72.8495)
    }
    
    for name, (lat, lon) in competing_stations.items():
        dists = get_distance(destinations['stop_lat'].values, destinations['stop_lon'].values, lat, lon)
        destinations = destinations[dists > 600]

    # Maximum Operational Radius (6km) to remove distant homes
    dists_from_origin = get_distance(destinations['stop_lat'].values, destinations['stop_lon'].values, station_lat, station_lon)
    destinations = destinations[dists_from_origin <= 6000]
    print(f"Filtered for local last-mile zones. Valid destinations remaining: {len(destinations):,}")

    if len(destinations) == 0:
        return None

    # DBSCAN Clustering
    print("Running DBSCAN Clustering...")
    coords = np.radians(destinations[['stop_lat', 'stop_lon']])
    epsilon = (300 / 1000.0) / 6371.0088 
    
    db = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
    destinations['hub_cluster_id'] = db.labels_

    valid_clusters = destinations[destinations['hub_cluster_id'] != -1]
    virtual_stops = []
    
    for hub_id, hub_data in valid_clusters.groupby('hub_cluster_id'):
        virtual_stops.append({
            'stop_id': f"Hub_{hub_id}",
            'lat': hub_data['stop_lat'].mean(),
            'lon': hub_data['stop_lon'].mean(),
            'unique_commuters': hub_data['device_aid'].nunique()
        })

    stops_df = pd.DataFrame(virtual_stops).sort_values(by='unique_commuters', ascending=False)
    stops_df.to_csv("smart_virtual_stops_checkpoint.csv", index=False)
    print(f"Generated {len(stops_df)} Smart Virtual Stops!")
    return stops_df

# ==========================================
# PHASE 3: GENETIC AI ROUTING (CVRP)
# ==========================================
def preprocess_demand(stops_df, capacity):
    processed_stops = []
    stop_idx = 1 
    for _, row in stops_df.iterrows():
        demand = row['unique_commuters']
        while demand > 0:
            load = min(demand, capacity)
            processed_stops.append({
                'matrix_idx': stop_idx,
                'stop_id': row['stop_id'],
                'lat': row['lat'], 'lon': row['lon'], 'load': load
            })
            demand -= load
            stop_idx += 1
    return processed_stops

def decode_chromosome(chromosome, stops, capacity, dist_matrix):
    routes, current_route, current_load, total_distance, last_node = [], [], 0, 0, 0
    for stop_idx in chromosome:
        stop = stops[stop_idx]
        if current_load + stop['load'] > capacity:
            total_distance += dist_matrix[last_node][0] 
            routes.append(current_route)
            current_route, current_load, last_node = [], 0, 0
            
        current_route.append(stop)
        current_load += stop['load']
        total_distance += dist_matrix[last_node][stop['matrix_idx']]
        last_node = stop['matrix_idx']
        
    if current_route:
        total_distance += dist_matrix[last_node][0]
        routes.append(current_route)
    return routes, total_distance

def order_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = parent1[a:b+1]
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child: p2_idx += 1
            child[i] = parent2[p2_idx]
    return child

def mutate(chromosome, mutation_rate=0.2):
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

def phase3_run_genetic_ai(stops_df, station_lat, station_lon, capacity=40, pop_size=100, generations=150):
    print("\n--- PHASE 3: GENETIC AI ROUTE EVOLUTION ---")
    processed_stops = preprocess_demand(stops_df, capacity)
    dist_matrix = build_distance_matrix(station_lat, station_lon, processed_stops)
    num_stops = len(processed_stops)
    
    population = [random.sample(range(num_stops), num_stops) for _ in range(pop_size)]
    best_chromosome, best_distance, best_routes = None, float('inf'), []
    
    print(f"Initiating Genetic Evolution ({generations} Generations)...")
    for gen in range(generations):
        scored_population = []
        for chromo in population:
            routes, dist = decode_chromosome(chromo, processed_stops, capacity, dist_matrix)
            scored_population.append((dist, chromo, routes))
            if dist < best_distance:
                best_distance, best_chromosome, best_routes = dist, chromo, routes
                
        scored_population.sort(key=lambda x: x[0])
        survivors = [item[1] for item in scored_population[:int(pop_size * 0.2)]]
        
        next_gen = copy.deepcopy(survivors)
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = mutate(order_crossover(p1, p2))
            next_gen.append(child)
        population = next_gen
        
        if (gen + 1) % 50 == 0:
            print(f"Generation {gen+1} | Best Route Distance: {round(best_distance / 1000, 2)} km")

    print(f"Evolution Complete! Dispatched {len(best_routes)} Optimized Buses.")
    return best_routes

# ==========================================
# VISUALIZATION (FINAL MAP)
# ==========================================
def draw_final_map(routes, station_lat, station_lon, output_filename="final_transit_masterplan.html"):
    print("\n--- FINALIZING: GENERATING DASHBOARD ---")
    m = folium.Map(location=[station_lat, station_lon], zoom_start=13, tiles='openstreetmap')

    folium.Marker(
        [station_lat, station_lon], 
        popup="<b>Origin Station (Bus Depot)</b>", 
        icon=folium.Icon(color='red', icon='train', prefix='fa')
    ).add_to(m)

    route_colors = ['#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

    for idx, route in enumerate(routes):
        color = route_colors[idx % len(route_colors)]
        bus_name = f"Route {idx + 1}"
        route_coords = [[station_lat, station_lon]]
        total_load = sum(stop['load'] for stop in route)
        
        for stop_num, stop in enumerate(route):
            route_coords.append([stop['lat'], stop['lon']])
            folium.CircleMarker(
                location=[stop['lat'], stop['lon']], radius=10, color=color,
                fill=True, fill_opacity=0.9, weight=2,
                tooltip=f"<b>{bus_name}</b><br>Boarding: {stop['load']} pax"
            ).add_to(m)

        route_coords.append([station_lat, station_lon]) 
        folium.PolyLine(
            locations=route_coords, color=color, weight=4, opacity=0.8,
            tooltip=f"{bus_name} (Total Load: {total_load} pax)"
        ).add_to(m)

    m.save(output_filename)
    print(f"Success! Masterplan saved to {output_filename}")


# ==========================================
# MASTER ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    RAW_STOPS_FILE = '../mumbai_multiday_stops_robust.csv'
    STATION_NAME = "Andheri Station"
    STATION_LAT = 19.1197
    STATION_LON = 72.8464
    
    # Catchment Bounding Box
    MIN_LAT, MAX_LAT = 19.1157, 19.1237
    MIN_LON, MAX_LON = 72.8404, 72.8524
    
    # AI Routing Parameters
    BUS_CAPACITY = 30
    GA_POPULATION = 100
    GA_GENERATIONS = 150

    print("==========================================")
    print("INITIALIZING AI URBAN MOBILITY ENGINE")
    print("==========================================")

    # Execute Pipeline
    commuters = phase1_extract_catchment(RAW_STOPS_FILE, STATION_NAME, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
    
    if commuters is not None:
        virtual_stops = phase2_generate_virtual_stops(commuters, STATION_LAT, STATION_LON)
        
        if virtual_stops is not None and len(virtual_stops) > 0:
            best_routes = phase3_run_genetic_ai(virtual_stops, STATION_LAT, STATION_LON, BUS_CAPACITY, GA_POPULATION, GA_GENERATIONS)
            draw_final_map(best_routes, STATION_LAT, STATION_LON)
        else:
            print("Pipeline halted: Not enough valid demand to generate routes.")
    else:
        print("Pipeline halted: No commuters found in catchment area.")