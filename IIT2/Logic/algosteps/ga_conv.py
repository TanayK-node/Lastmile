import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import IIT2.Logic.Final.osmnx_router as oxr
import route2 as re

def nearest_neighbour_baseline(G, stops, capacity, dist_matrix, station_idx=0):
    """
    The "Dumb" Baseline: Always drive to the absolute closest unvisited stop.
    Used to prove the Genetic Algorithm's intelligence.
    """
    unvisited = stops.copy()
    routes = []
    current_route = []
    current_load = 0
    total_dist = 0
    total_ride = 0
    last_node = station_idx
    
    while unvisited:
        # Find the closest unvisited stop
        closest_stop = None
        min_dist = float('inf')
        
        for stop in unvisited:
            if dist_matrix[last_node][stop['matrix_idx']] < min_dist:
                min_dist = dist_matrix[last_node][stop['matrix_idx']]
                closest_stop = stop
                
        # If capacity exceeded, return to depot and dispatch new bus
        if current_load + closest_stop['load'] > capacity:
            dist_to_depot = dist_matrix[last_node][0]
            total_dist += dist_to_depot
            total_ride += (current_load * dist_to_depot)
            
            routes.append(current_route)
            current_route = []
            current_load = 0
            last_node = station_idx
            continue # Try adding the stop again to the fresh bus
            
        # Add to current route
        current_route.append(closest_stop)
        current_load += closest_stop['load']
        total_dist += min_dist
        total_ride += (current_load * min_dist)
        last_node = closest_stop['matrix_idx']
        unvisited.remove(closest_stop)
        
    # Return final bus to depot
    if current_route:
        dist_to_depot = dist_matrix[last_node][0]
        total_dist += dist_to_depot
        total_ride += (current_load * dist_to_depot)
        routes.append(current_route)
        
    return total_dist, total_ride

def track_ga_convergence(G, stops_df, station_lat, station_lon, alpha=0.5, capacity=10, pop_size=100, generations=150):
    """
    Modified GA run that tracks the fitness score at every single generation.
    """
    processed_stops = re.preprocess_demand(stops_df, capacity)
    dist_matrix = oxr.build_osmnx_distance_matrix(G, station_lat, station_lon, processed_stops)
    num_stops = len(processed_stops)
    
    population = [random.sample(range(num_stops), num_stops) for _ in range(pop_size)]
    
    best_weighted_score = float('inf')
    convergence_history = [] # The new tracking array
    
    base_mutation = 0.10
    spike_mutation = 0.50
    stagnation_limit = 10
    stagnation_counter = 0
    current_mutation = base_mutation
    
    print("Tracking Evolution across 150 Generations...")
    for gen in range(generations):
        scored_population = []
        gen_improved = False
        
        for chromo in population:
            routes, dist, ride = re.decode_chromosome_mo(chromo, processed_stops, capacity, dist_matrix)
            weighted_score = (alpha * dist) + ((1 - alpha) * (ride / capacity))
            scored_population.append((weighted_score, chromo))
            
            if weighted_score < best_weighted_score:
                best_weighted_score = weighted_score
                gen_improved = True
                
        # --- THE ADAPTIVE LOGIC ---
        if gen_improved:
            stagnation_counter = 0
            current_mutation = base_mutation
        else:
            stagnation_counter += 1
            
        if stagnation_counter >= stagnation_limit:
            current_mutation = spike_mutation
            
        convergence_history.append(best_weighted_score)
        
        scored_population.sort(key=lambda x: x[0])
        elite_count = max(1, int(pop_size * 0.2))
        survivors = [item[1] for item in scored_population[:elite_count]]
        
        next_gen = copy.deepcopy(survivors)
        while len(next_gen) < pop_size:
            if len(survivors) >= 2:
                p1, p2 = random.sample(survivors, 2)
            else:
                p1 = p2 = survivors[0]
            next_gen.append(re.mutate(re.order_crossover(p1, p2), current_mutation))
        population = next_gen

    # Calculate the baseline for comparison
    nn_dist, nn_ride = nearest_neighbour_baseline(G, processed_stops, capacity, dist_matrix)
    nn_score = (alpha * nn_dist) + ((1 - alpha) * (nn_ride / capacity))
    
    return convergence_history, nn_score

if __name__ == "__main__":
    print("Loading Graph for Convergence Tracking...")
    G = oxr.load_or_download_mumbai_graph()
    
    # Extract one station's demand
    feeder_df = pd.read_csv("./demand_matrices/first_mile_feeder_demand.csv")
    station_feeder = feeder_df[feeder_df['station'] == "Andheri"]
    s_lat, s_lon = 19.1197, 72.8464
    stops = re.generate_virtual_stops(station_feeder, s_lat, s_lon)
    
    ga_history, nn_baseline = track_ga_convergence(G, stops, s_lat, s_lon)
    
    # --- PLOTTING THE ACADEMIC GRAPH ---
    plt.figure(figsize=(10, 6))
    
    # Plot the GA learning curve
    plt.plot(range(150), ga_history, label='Adaptive Genetic Algorithm (AGA)', color='blue', linewidth=2)
    
    # Plot the Baseline as a flat threshold
    plt.axhline(y=nn_baseline, color='red', linestyle='--', label='Nearest-Neighbour Baseline', linewidth=2)
    
    # Find exact point of convergence
    final_score = ga_history[-1]
    convergence_gen = next(i for i, v in enumerate(ga_history) if v <= final_score * 1.01)
    
    plt.plot(convergence_gen, final_score, 'go', markersize=8, label=f'Convergence Point (Gen {convergence_gen})')
    
    plt.title('Algorithm Convergence: AGA vs Nearest-Neighbour Baseline', fontsize=14)
    plt.xlabel('Evolutionary Generations', fontsize=12)
    plt.ylabel('Multi-Objective Fitness Score (Lower is Better)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.savefig('algorithm_convergence.png', dpi=300, bbox_inches='tight')
    print(f"Convergence graph saved! Algorithm stabilized at Generation {convergence_gen}.")