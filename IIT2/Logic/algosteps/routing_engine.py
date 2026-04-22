import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
import random
import copy
import os

# ==========================================
# 1. CORE MATH & DISTANCE
# ==========================================
def get_distance(lat_array, lon_array, target_lat, target_lon):
    R = 6371000  
    phi1, phi2 = np.radians(lat_array), np.radians(target_lat)
    dphi, dlambda = np.radians(target_lat - lat_array), np.radians(target_lon - lon_array)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def build_distance_matrix(station_lat, station_lon, stops):
    all_points = [{'lat': station_lat, 'lon': station_lon}] + stops
    n = len(all_points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = get_distance(all_points[i]['lat'], all_points[i]['lon'], 
                                            all_points[j]['lat'], all_points[j]['lon'])
    return matrix

# ==========================================
# 2. AI VIRTUAL STOPS (DBSCAN)
# ==========================================
def generate_virtual_stops(demand_df, station_lat, station_lon):
    if len(demand_df) < 5: # Need minimum points for clustering
        return None

    coords = np.radians(demand_df[['lat', 'lon']])
    epsilon = (300 / 1000.0) / 6371.0088 
    
    # DBSCAN clusters the demand into high-density virtual bus stops
    db = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
    demand_df = demand_df.copy()
    demand_df['hub_cluster_id'] = db.labels_

    valid_clusters = demand_df[demand_df['hub_cluster_id'] != -1]
    virtual_stops = []
    
    for hub_id, hub_data in valid_clusters.groupby('hub_cluster_id'):
        virtual_stops.append({
            'stop_id': f"Hub_{hub_id}",
            'lat': hub_data['lat'].mean(),
            'lon': hub_data['lon'].mean(),
            'unique_commuters': hub_data['device_aid'].nunique()
        })

    return pd.DataFrame(virtual_stops).sort_values(by='unique_commuters', ascending=False)

# ==========================================
# 3. GENETIC AI ROUTING (CVRP)
# ==========================================
def preprocess_demand(stops_df, capacity):
    processed_stops = []
    stop_idx = 1 
    for _, row in stops_df.iterrows():
        demand = row['unique_commuters']
        while demand > 0:
            load = min(demand, capacity)
            processed_stops.append({
                'matrix_idx': stop_idx, 'stop_id': row['stop_id'],
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

def run_genetic_ai(stops_df, station_lat, station_lon, capacity=30, pop_size=100, generations=150):
    processed_stops = preprocess_demand(stops_df, capacity)
    dist_matrix = build_distance_matrix(station_lat, station_lon, processed_stops)
    num_stops = len(processed_stops)
    
    population = [random.sample(range(num_stops), num_stops) for _ in range(pop_size)]
    best_chromosome, best_distance, best_routes = None, float('inf'), []
    
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
            next_gen.append(mutate(order_crossover(p1, p2)))
        population = next_gen

    return best_routes, best_distance

# ==========================================
# 4. DASHBOARD VISUALIZATION
# ==========================================
def draw_final_map(routes, station_name, station_lat, station_lon, direction, output_filename):
    m = folium.Map(location=[station_lat, station_lon], zoom_start=13, tiles='CartoDB positron')

    # Station Marker
    folium.Marker(
        [station_lat, station_lon], 
        popup=f"<b>{station_name} (Bus Depot)</b>", 
        icon=folium.Icon(color='black', icon='train', prefix='fa')
    ).add_to(m)

    route_colors = ['#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#800000', '#000075']

    for idx, route in enumerate(routes):
        color = route_colors[idx % len(route_colors)]
        bus_name = f"{direction} Route {idx + 1}"
        route_coords = [[station_lat, station_lon]]
        total_load = sum(stop['load'] for stop in route)
        
        for stop in route:
            route_coords.append([stop['lat'], stop['lon']])
            folium.CircleMarker(
                location=[stop['lat'], stop['lon']], radius=7, color=color,
                fill=True, fill_opacity=0.9, weight=2,
                tooltip=f"<b>{bus_name}</b><br>Demand: {stop['load']} pax"
            ).add_to(m)

        route_coords.append([station_lat, station_lon]) 
        
        folium.PolyLine(
            locations=route_coords, color=color, weight=4, opacity=0.8,
            tooltip=f"{bus_name} (Total Load: {total_load} pax)"
        ).add_to(m)

    m.save(output_filename)

# ==========================================
# 5. THE MULTI-DEPOT ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("INITIALIZING BI-DIRECTIONAL AI ROUTING ENGINE")
    print("==========================================\n")

    # The mathematically proven hyperparameters
    BUS_CAPACITY = 30
    GA_POPULATION = 100
    GA_GENERATIONS = 150

    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }

    # Load the verified matrices from Step 1
    feeder_df = pd.read_csv("./demand_matrices/first_mile_feeder_demand.csv")
    dispersal_df = pd.read_csv("./demand_matrices/last_mile_dispersal_demand.csv")

    output_dir = "./final_fleet_maps"
    os.makedirs(output_dir, exist_ok=True)

    total_buses_dispatched = 0
    total_system_mileage = 0

    for station_name, (s_lat, s_lon) in STATIONS.items():
        print(f"--- Processing {station_name} Station ---")
        
        # --- 1. MORNING FEEDER FLEET ---
        station_feeder = feeder_df[feeder_df['station'] == station_name]
        feeder_stops = generate_virtual_stops(station_feeder, s_lat, s_lon)
        
        if feeder_stops is not None and len(feeder_stops) > 0:
            routes, dist = run_genetic_ai(feeder_stops, s_lat, s_lon, BUS_CAPACITY, GA_POPULATION, GA_GENERATIONS)
            total_buses_dispatched += len(routes)
            total_system_mileage += dist
            
            map_name = os.path.join(output_dir, f"{station_name}_Feeder_Fleet.html")
            draw_final_map(routes, station_name, s_lat, s_lon, "Morning Feeder", map_name)
            print(f"  [+] Morning Feeder: Dispatched {len(routes)} buses ({round(dist/1000, 1)} km). Map saved.")
        else:
            print("  [-] Morning Feeder: Insufficient demand.")

        # --- 2. EVENING DISPERSAL FLEET ---
        station_dispersal = dispersal_df[dispersal_df['station'] == station_name]
        dispersal_stops = generate_virtual_stops(station_dispersal, s_lat, s_lon)
        
        if dispersal_stops is not None and len(dispersal_stops) > 0:
            routes, dist = run_genetic_ai(dispersal_stops, s_lat, s_lon, BUS_CAPACITY, GA_POPULATION, GA_GENERATIONS)
            total_buses_dispatched += len(routes)
            total_system_mileage += dist
            
            map_name = os.path.join(output_dir, f"{station_name}_Dispersal_Fleet.html")
            draw_final_map(routes, station_name, s_lat, s_lon, "Evening Dispersal", map_name)
            print(f"  [+] Evening Dispersal: Dispatched {len(routes)} buses ({round(dist/1000, 1)} km). Map saved.")
        else:
            print("  [-] Evening Dispersal: Insufficient demand.")
            
        print("") # Spacing

    print("==========================================")
    print("CITY-WIDE OPTIMIZATION COMPLETE")
    print(f"Total Buses Dispatched: {total_buses_dispatched}")
    print(f"Total System Mileage: {round(total_system_mileage/1000, 2)} km")
    print("==========================================")
    import json
    metrics_export = {
        "total_buses": total_buses_dispatched,
        "total_mileage_km": total_system_mileage / 1000
    }
    with open("routing_summary.json", "w") as f:
        json.dump(metrics_export, f)
    print("Exported routing metrics for academic evaluation.")