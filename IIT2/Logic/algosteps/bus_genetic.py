import pandas as pd
import numpy as np
import folium
import random
import copy

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def build_distance_matrix(station_lat, station_lon, stops):
    """Precomputes all distances to make the Genetic Algorithm lightning fast."""
    all_points = [{'lat': station_lat, 'lon': station_lon}] + stops
    n = len(all_points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine(all_points[i]['lat'], all_points[i]['lon'], 
                                         all_points[j]['lat'], all_points[j]['lon'])
    return matrix

def preprocess_demand(stops_df, capacity):
    """Splits stops with demand > capacity into multiple distinct boarding events."""
    processed_stops = []
    stop_idx = 1 # 0 is reserved for the station
    
    for _, row in stops_df.iterrows():
        demand = row['unique_commuters']
        while demand > 0:
            load = min(demand, capacity)
            processed_stops.append({
                'matrix_idx': stop_idx,
                'stop_id': row['stop_id'],
                'lat': row['lat'],
                'lon': row['lon'],
                'load': load
            })
            demand -= load
            stop_idx += 1
    return processed_stops

# ==========================================
# GENETIC ALGORITHM CORE
# ==========================================
def decode_chromosome(chromosome, stops, capacity, dist_matrix):
    """Converts a sequence of stops into valid bus routes and calculates total distance."""
    routes = []
    current_route = []
    current_load = 0
    total_distance = 0
    
    # 0 represents the Train Station
    last_node = 0 
    
    for stop_idx in chromosome:
        stop = stops[stop_idx]
        
        # If this stop exceeds the current bus capacity, send the bus back to station and start a new one
        if current_load + stop['load'] > capacity:
            total_distance += dist_matrix[last_node][0] # Drive back to station
            routes.append(current_route)
            
            # Start new bus
            current_route = []
            current_load = 0
            last_node = 0
            
        # Add stop to current bus
        current_route.append(stop)
        current_load += stop['load']
        total_distance += dist_matrix[last_node][stop['matrix_idx']]
        last_node = stop['matrix_idx']
        
    # Finish the very last route
    if current_route:
        total_distance += dist_matrix[last_node][0]
        routes.append(current_route)
        
    return routes, total_distance

def order_crossover(parent1, parent2):
    """Standard Genetic Algorithm OX1 crossover for permutations."""
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[a:b+1] = parent1[a:b+1]
    
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
    return child

def mutate(chromosome, mutation_rate=0.1):
    """Randomly swaps two stops to discover new shortcuts."""
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

def run_genetic_algorithm(stops_file, station_lat, station_lon, capacity=40, pop_size=100, generations=150):
    print(f"Loading Virtual Stops from {stops_file}...")
    df = pd.read_csv(stops_file)
    
    print("\nPre-processing demand and building spatial matrix...")
    processed_stops = preprocess_demand(df, capacity)
    dist_matrix = build_distance_matrix(station_lat, station_lon, processed_stops)
    
    num_stops = len(processed_stops)
    
    # 1. INITIALIZATION: Generate random routes
    population = [random.sample(range(num_stops), num_stops) for _ in range(pop_size)]
    best_chromosome = None
    best_distance = float('inf')
    best_routes = []
    
    print(f"\nInitiating Genetic Evolution ({generations} Generations)...")
    
    for gen in range(generations):
        # 2. FITNESS EVALUATION
        scored_population = []
        for chromo in population:
            routes, dist = decode_chromosome(chromo, processed_stops, capacity, dist_matrix)
            scored_population.append((dist, chromo, routes))
            
            if dist < best_distance:
                best_distance = dist
                best_chromosome = chromo
                best_routes = routes
                
        # Sort by shortest distance
        scored_population.sort(key=lambda x: x[0])
        
        # 3. SELECTION (Keep top 20%)
        survivors = [item[1] for item in scored_population[:int(pop_size * 0.2)]]
        
        # 4. CROSSOVER (Breed next generation)
        next_gen = copy.deepcopy(survivors)
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = order_crossover(p1, p2)
            # 5. MUTATION
            child = mutate(child, mutation_rate=0.2)
            next_gen.append(child)
            
        population = next_gen
        
        if (gen + 1) % 25 == 0:
            print(f"Generation {gen+1} | Best Distance: {round(best_distance / 1000, 2)} km")

    print("\n" + "="*40)
    print("EVOLUTION COMPLETE")
    print(f"Total Buses Dispatched: {len(best_routes)}")
    print(f"Optimized Fleet Mileage: {round(best_distance / 1000, 2)} km")
    print("="*40 + "\n")

    return best_routes, station_lat, station_lon

def map_routes(routes, station_lat, station_lon):
    """Draws the final evolved routes on a Folium Map."""
    print("Mapping genetically optimized routes...")
    m = folium.Map(location=[station_lat, station_lon], zoom_start=14, tiles='CartoDB positron')

    folium.Marker(
        [station_lat, station_lon], 
        popup="<b>Andheri Station (Origin)</b>", 
        icon=folium.Icon(color='red', icon='train', prefix='fa')
    ).add_to(m)

    route_colors = ['#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

    for idx, route in enumerate(routes):
        color = route_colors[idx % len(route_colors)]
        bus_name = f"Gen-AI Bus {idx + 1}"
        route_coords = [[station_lat, station_lon]]
        total_load = sum(stop['load'] for stop in route)
        
        for stop_num, stop in enumerate(route):
            route_coords.append([stop['lat'], stop['lon']])
            
            folium.Marker(
                location=[stop['lat'], stop['lon']],
                popup=f"<b>{bus_name} - Stop {stop_num + 1}</b><br>{stop['stop_id']}<br>Boarded: {stop['load']} pax",
                icon=folium.Icon(color='black', icon_color=color, icon='bus', prefix='fa')
            ).add_to(m)

        # Draw the physical path
        route_coords.append([station_lat, station_lon]) # Bus returns to station
        
        folium.PolyLine(
            locations=route_coords,
            color=color, weight=4, opacity=0.8,
            tooltip=f"{bus_name} (Load: {total_load} pax)"
        ).add_to(m)

    m.save("genetic_fleet_routes.html")
    print("Success! Final map saved as genetic_fleet_routes.html")

# ==========================================
# HOW TO USE
# ==========================================
STOPS_FILE = 'smart_virtual_stops.csv'
STATION_LAT = 19.1197
STATION_LON = 72.8464

# Run the Genetic Algorithm!
# Note: You can increase 'generations=150' to 'generations=500' if you want the AI to search longer for a better route!
final_routes, s_lat, s_lon = run_genetic_algorithm(STOPS_FILE, STATION_LAT, STATION_LON, capacity=40, pop_size=100, generations=150)
map_routes(final_routes, s_lat, s_lon)