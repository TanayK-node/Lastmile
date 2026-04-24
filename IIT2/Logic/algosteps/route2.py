import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
import random
import copy
import os
import osmnx as ox
import networkx as nx
import IIT2.Logic.Final.osmnx_router as oxr # Make sure your filename is osmnx_router.py
import eval2 as ev2 # For the evaluation metrics after routing

# ==========================================
# 1. AI VIRTUAL STOPS (DBSCAN)
# ==========================================
def generate_virtual_stops(demand_df, station_lat, station_lon):
    if len(demand_df) < 5: 
        return None

    coords = np.radians(demand_df[['lat', 'lon']])
    epsilon = (300 / 1000.0) / 6371.0088 
    
    db = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
    demand_df = demand_df.copy()
    demand_df['hub_cluster_id'] = db.labels_

    valid_clusters = demand_df[demand_df['hub_cluster_id'] != -1]
    if valid_clusters.empty:
        return None

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
# 2. GENETIC AI ROUTING (CVRP)
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

def decode_chromosome_mo(chromosome, stops, capacity, dist_matrix):
    """
    MULTI-OBJECTIVE DECODER
    Calculates both total fleet mileage (Operator Cost) 
    and total passenger-meters (Commuter Cost).
    """
    routes, current_route = [], []
    current_load, total_distance, total_passenger_ride_dist = 0, 0, 0
    last_node = 0

    for stop_idx in chromosome:
        stop = stops[stop_idx]
        
        # If adding this stop exceeds capacity, return to depot first
        if current_load + stop['load'] > capacity:
            dist_to_depot = dist_matrix[last_node][0]
            total_distance += dist_to_depot
            # The passengers currently trapped on the bus endure this distance
            total_passenger_ride_dist += (current_load * dist_to_depot) 
            
            routes.append(current_route)
            current_route, current_load, last_node = [], 0, 0
            
        # Drive to the next stop
        dist_to_stop = dist_matrix[last_node][stop['matrix_idx']]
        total_distance += dist_to_stop
        # Everyone currently on the bus feels this travel time
        total_passenger_ride_dist += (current_load * dist_to_stop)
        
        # Pick up the new passengers
        current_route.append(stop)
        current_load += stop['load']
        last_node = stop['matrix_idx']
        
    # Final return to depot for the last bus
    if current_route:
        dist_to_depot = dist_matrix[last_node][0]
        total_distance += dist_to_depot
        total_passenger_ride_dist += (current_load * dist_to_depot)
        routes.append(current_route)
        
    return routes, total_distance, total_passenger_ride_dist

def order_crossover(parent1, parent2):
    size = len(parent1)
    if size < 2:
        return parent1.copy()

    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = parent1[a:b+1]
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child: p2_idx += 1
            child[i] = parent2[p2_idx]
    return child

def mutate(chromosome, current_mutation_rate):
    """
    Applies random swap mutation based on the DYNAMIC rate provided by the AGA.
    """
    if len(chromosome) < 2:
        return chromosome

    if random.random() < current_mutation_rate:
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

# NOTE: Passed 'G' into the function parameters here
def run_genetic_ai(G, stops_df, station_lat, station_lon, alpha=0.5, capacity=30, pop_size=100, generations=150):
    """
    ADAPTIVE MULTI-OBJECTIVE GENETIC ALGORITHM (AGA)
    Features dynamic mutation scaling to prevent premature convergence.
    """
    processed_stops = preprocess_demand(stops_df, capacity)
    dist_matrix = oxr.build_osmnx_distance_matrix(G, station_lat, station_lon, processed_stops)
    num_stops = len(processed_stops)
    
    population = [random.sample(range(num_stops), num_stops) for _ in range(pop_size)]
    
    best_weighted_score = float('inf')
    best_distance, best_ride_dist, best_routes = 0, 0, []
    
    # --- AGA PARAMETERS ---
    base_mutation_rate = 0.10
    spike_mutation_rate = 0.50
    stagnation_limit = 10
    stagnation_counter = 0
    current_mutation_rate = base_mutation_rate
    
    for gen in range(generations):
        scored_population = []
        generation_improved = False
        
        for chromo in population:
            routes, dist, ride_dist = decode_chromosome_mo(chromo, processed_stops, capacity, dist_matrix)
            
            normalized_dist = dist
            normalized_ride = ride_dist / capacity 
            weighted_score = (alpha * normalized_dist) + ((1 - alpha) * normalized_ride)
            
            scored_population.append((weighted_score, chromo, routes, dist, ride_dist))
            
            if weighted_score < best_weighted_score:
                best_weighted_score = weighted_score
                best_distance = dist
                best_ride_dist = ride_dist
                best_routes = routes
                generation_improved = True
                
        # --- AGA STAGNATION LOGIC ---
        if generation_improved:
            stagnation_counter = 0
            current_mutation_rate = base_mutation_rate # Cool down
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= stagnation_limit:
            current_mutation_rate = spike_mutation_rate # Force exploration!
            # We don't print every time, but this happens silently in the background
                
        # Elitism and Crossover
        scored_population.sort(key=lambda x: x[0])
        elite_count = max(1, int(pop_size * 0.2))
        survivors = [item[1] for item in scored_population[:elite_count]]
        
        next_gen = copy.deepcopy(survivors)
        while len(next_gen) < pop_size:
            if len(survivors) >= 2:
                p1, p2 = random.sample(survivors, 2)
            else:
                p1 = p2 = survivors[0]
            # Pass the DYNAMIC mutation rate to the offspring
            next_gen.append(mutate(order_crossover(p1, p2), current_mutation_rate))
            
        population = next_gen

    return best_routes, best_distance, best_ride_dist

# ==========================================
# 3. DASHBOARD VISUALIZATION (ROAD-MATCHED)
# ==========================================
def get_street_path(G, lat1, lon1, lat2, lon2):
    """Fetches the true turn-by-turn road geometry between two points."""
    try:
        orig_node = ox.distance.nearest_nodes(G, X=lon1, Y=lat1)
        dest_node = ox.distance.nearest_nodes(G, X=lon2, Y=lat2)
        route = nx.shortest_path(G, orig_node, dest_node, weight='length')
        # Extract lat/lon for every intersection on the path
        return [[G.nodes[n]['y'], G.nodes[n]['x']] for n in route]
    except Exception:
        # Fallback to straight line if a road is totally blocked
        return [[lat1, lon1], [lat2, lon2]]

def draw_final_map(G, routes, station_name, station_lat, station_lon, direction, output_filename):
    print(f"  * Tracing {len(routes)} bus routes onto physical street maps...")
    m = folium.Map(location=[station_lat, station_lon], zoom_start=14, tiles='CartoDB positron')

    folium.Marker(
        [station_lat, station_lon], 
        popup=f"<b>{station_name} (Bus Depot)</b>", 
        icon=folium.Icon(color='black', icon='train', prefix='fa')
    ).add_to(m)

    route_colors = ['#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#800000', '#000075']

    for idx, route in enumerate(routes):
        color = route_colors[idx % len(route_colors)]
        bus_name = f"{direction} Route {idx + 1}"
        total_load = sum(stop['load'] for stop in route)
        
        full_route_coords = []
        
        # 1. Path from Station to First Stop
        first_stop = route[0]
        full_route_coords.extend(get_street_path(G, station_lat, station_lon, first_stop['lat'], first_stop['lon']))
        
        # 2. Path between all Virtual Stops
        for i in range(len(route) - 1):
            stop1, stop2 = route[i], route[i+1]
            full_route_coords.extend(get_street_path(G, stop1['lat'], stop1['lon'], stop2['lat'], stop2['lon']))
            
        # 3. Path from Last Stop back to Station
        last_stop = route[-1]
        full_route_coords.extend(get_street_path(G, last_stop['lat'], last_stop['lon'], station_lat, station_lon))

        # Draw the virtual stops as dots
        for stop in route:
            folium.CircleMarker(
                location=[stop['lat'], stop['lon']], radius=7, color=color,
                fill=True, fill_opacity=0.9, weight=2,
                tooltip=f"<b>{bus_name}</b><br>Demand: {stop['load']} pax"
            ).add_to(m)

        # Draw the true physical path
        folium.PolyLine(
            locations=full_route_coords, color=color, weight=5, opacity=0.8,
            tooltip=f"{bus_name} (Total Load: {total_load} pax)"
        ).add_to(m)

    m.save(output_filename)

# ==========================================
# 4. THE MULTI-DEPOT ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("INITIALIZING 4D BI-DIRECTIONAL AI ROUTING ENGINE")
    print("==========================================\n")

    BUS_CAPACITY = 20
    GA_POPULATION = 100
    GA_GENERATIONS = 150

    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }

    # LOAD ALL 4 TEMPORAL MATRICES
    try:
        am_feeder_df = pd.read_csv("./demand_matrices/AM_feeder_demand.csv")
        am_dispersal_df = pd.read_csv("./demand_matrices/AM_dispersal_demand.csv")
        pm_feeder_df = pd.read_csv("./demand_matrices/PM_feeder_demand.csv")
        pm_dispersal_df = pd.read_csv("./demand_matrices/PM_dispersal_demand.csv")
    except FileNotFoundError:
        print("Error: Could not find the 4D demand matrices. Run chain_extractor.py first!")
        exit()

    output_dir = "./final_fleet_maps"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Mumbai Street Network...")
    G = oxr.load_or_download_mumbai_graph()
    total_buses_dispatched = 0
    total_system_mileage = 0

    # Accumulators for citywide evaluation
    all_routes = []
    all_demand_dfs = []
    all_virtual_stops_list = []

    for station_name, (s_lat, s_lon) in STATIONS.items():
        print(f"\n--- Processing {station_name} Station ---")
        
        # Package the 4 scenarios to loop through cleanly
        scenarios = [
            ("AM_Feeder", am_feeder_df),
            ("AM_Dispersal", am_dispersal_df),
            ("PM_Feeder", pm_feeder_df),
            ("PM_Dispersal", pm_dispersal_df)
        ]

        for scenario_name, scenario_df in scenarios:
            # Isolate the demand for this specific station and time window
            station_demand = scenario_df[scenario_df['station'] == station_name]
            virtual_stops = generate_virtual_stops(station_demand, s_lat, s_lon)
            
            if virtual_stops is not None and len(virtual_stops) > 0:
                total_pax = virtual_stops['unique_commuters'].sum()
                print(f"  -> Total {scenario_name} Demand: {total_pax} passengers")
                
                # Run the AI Optimization
                routes, dist, ride = run_genetic_ai(
                    G, virtual_stops, s_lat, s_lon, 
                    alpha=0.5, 
                    capacity=BUS_CAPACITY, 
                    pop_size=GA_POPULATION, 
                    generations=GA_GENERATIONS
                )
                
                total_buses_dispatched += len(routes)
                total_system_mileage += dist
                
                # Save the Map dynamically using the scenario name
                map_name = os.path.join(output_dir, f"{station_name}_{scenario_name}_Fleet.html")
                draw_final_map(G, routes, station_name, s_lat, s_lon, scenario_name.replace("_", " "), map_name)
                print(f"  [+] {scenario_name}: Dispatched {len(routes)} buses ({round(dist/1000, 1)} km). Map saved.")

                # Add to evaluators
                all_routes.extend(routes)
                all_demand_dfs.append(station_demand)
                all_virtual_stops_list.append(virtual_stops)
            else:
                print(f"  [-] {scenario_name}: Insufficient demand.")

    # Run combined citywide evaluation with all stations and all 4 time blocks
    if all_routes and all_demand_dfs and all_virtual_stops_list:
        combined_demand = pd.concat(all_demand_dfs, ignore_index=True)
        combined_stops = pd.concat(all_virtual_stops_list, ignore_index=True)
        
        # Use center point of all stations for reference
        first_station = list(STATIONS.values())[0]
        
        print("\nCalculating City-Wide Evaluation Metrics...")
        eval_df = ev2.evaluate_fleet_performance(
            G, all_routes, combined_demand, combined_stops, 
            first_station[0], first_station[1], BUS_CAPACITY
        )
        print("\n" + eval_df.to_markdown(index=False) + "\n")
        
    print("\n==========================================")
    print("4D CITY-WIDE OPTIMIZATION COMPLETE")
    print(f"Total Minibuses Dispatched: {total_buses_dispatched}")
    print(f"Total System Mileage: {round(total_system_mileage/1000, 2)} km")
    print("==========================================")