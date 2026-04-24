import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import os
import random
import copy

# Import only the safe, static modules from your project
import osmnx_router as oxr
import tsp_sequencer as tsp

# ==========================================
# INTERNAL CLUSTERER (Protects your main code)
# ==========================================
def temp_calculate_fitness(chromosome, virtual_stops_list, time_matrix, max_cluster_time):
    active_zones = set(chromosome)
    fleet_size = len(active_zones)
    fleet_penalty = fleet_size * 500 
    
    compactness_cost = 0
    time_violation_penalty = 0
    
    zones = {z: [] for z in active_zones}
    for idx, zone_id in enumerate(chromosome):
        matrix_idx = virtual_stops_list[idx]['matrix_idx']
        zones[zone_id].append(matrix_idx)
        
    for zone_id, indices in zones.items():
        if len(indices) > 1:
            zone_pairwise_time = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    time_cost = time_matrix[indices[i]][indices[j]]
                    zone_pairwise_time += time_cost
                    compactness_cost += time_cost
            
            if zone_pairwise_time > max_cluster_time:
                time_violation_penalty += (zone_pairwise_time - max_cluster_time) * 10000
                
    return fleet_penalty + compactness_cost + time_violation_penalty

def temp_crossover(parent1, parent2):
    return [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]

def temp_mutate(chromosome, max_zones, mutation_rate=0.15):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(0, max_zones - 1)
    return chromosome

def standalone_cluster_hubs(virtual_stops_df, time_matrix, max_cluster_time, pop_size=30, generations=30):
    """A self-contained GA that doesn't rely on your main route_clusterer.py"""
    if virtual_stops_df is None or virtual_stops_df.empty:
        return None
        
    virtual_stops = virtual_stops_df.copy()
    virtual_stops_list = virtual_stops.to_dict('records')
    num_stops = len(virtual_stops)
    
    if num_stops <= 2:
        virtual_stops['route_zone_id'] = 0
        return virtual_stops

    max_possible_zones = max(2, int(num_stops * 0.8))
    
    population = [[random.randint(0, max_possible_zones - 1) for _ in range(num_stops)] for _ in range(pop_size)]
    best_chromosome = []
    best_fitness = float('inf')
    
    for _ in range(generations):
        scored = [(temp_calculate_fitness(c, virtual_stops_list, time_matrix, max_cluster_time), c) for c in population]
        scored.sort(key=lambda x: x[0])
        
        if scored[0][0] < best_fitness:
            best_fitness = scored[0][0]
            best_chromosome = scored[0][1]
            
        survivors = [item[1] for item in scored[:int(pop_size * 0.2)]]
        next_gen = copy.deepcopy(survivors)
        
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            next_gen.append(temp_mutate(temp_crossover(p1, p2), max_possible_zones))
            
        population = next_gen

    virtual_stops['route_zone_id'] = best_chromosome
    return virtual_stops

# ==========================================
# DATA & ROUTING UTILS
# ==========================================
def generate_virtual_stops(demand_df):
    if len(demand_df) < 5: return None
    coords = np.radians(demand_df[['lat', 'lon']])
    db = DBSCAN(eps=(300 / 1000.0) / 6371.0088, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
    demand_df = demand_df.copy()
    demand_df['hub_cluster_id'] = db.labels_
    
    virtual_stops = []
    stop_idx = 1
    for hub_id, hub_data in demand_df[demand_df['hub_cluster_id'] != -1].groupby('hub_cluster_id'):
        virtual_stops.append({
            'matrix_idx': stop_idx, 'stop_id': f"Hub_{hub_id}",
            'lat': hub_data['lat'].mean(), 'lon': hub_data['lon'].mean(),
            'unique_commuters': hub_data['device_aid'].nunique()
        })
        stop_idx += 1
    return pd.DataFrame(virtual_stops).sort_values(by='unique_commuters', ascending=False)

# ==========================================
# MAIN SENSITIVITY SWEEP
# ==========================================
def run_standalone_analysis():
    print("==========================================")
    print("RUNNING STANDALONE PARETO SWEEP")
    print("==========================================\n")
    
    # We will test just Andheri and Bandra to keep the script fast
    STATIONS = {"Andheri": (19.1197, 72.8464), "Bandra": (19.0544, 72.8402)} 
    
    try:
        am_feeder = pd.read_csv("./demand_matrices/AM_feeder_demand.csv")
    except FileNotFoundError:
        print("Error: Demand CSV not found. Make sure you are in the main project folder.")
        return

    time_limits_to_test = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    results = []
    
    for time_limit in time_limits_to_test:
        print(f"--- Testing Time Limit: {time_limit} Mins ---")
        
        sweep_times, sweep_distances = [], []
        
        for station_name, (s_lat, s_lon) in STATIONS.items():
            G = oxr.load_or_download_station_graph(station_name, s_lat, s_lon)
            
            # The safety fix to ensure the graph doesn't break
            try:
                G = oxr.ox.utils_graph.get_largest_component(G, strongly=True)
            except AttributeError:
                pass # Failsafe if using an older OSMnx version
            
            station_demand = am_feeder[am_feeder['station'] == station_name]
            virtual_stops = generate_virtual_stops(station_demand)
            
            if virtual_stops is not None and len(virtual_stops) > 0:
                dist_matrix, time_matrix, _ = oxr.build_osmnx_distance_matrix(G, s_lat, s_lon, virtual_stops.to_dict('records'))
                
                # USING THE ISOLATED CLUSTERER
                zoned_stops = standalone_cluster_hubs(virtual_stops, time_matrix, max_cluster_time=time_limit)
                
                for zone_id, zone_data in zoned_stops.groupby('route_zone_id'):
                    _, r_time, r_dist = tsp.optimize_cluster_route(zone_data.to_dict('records'), time_matrix, dist_matrix, depot_idx=0, pop_size=30, generations=30)
                    sweep_times.append(r_time)
                    sweep_distances.append(r_dist)
                        
        fleet_size = len(sweep_times)
        avg_time = np.mean(sweep_times) if fleet_size > 0 else 0
        est_commuter_ride_time = (avg_time * 0.6) + 2.0 if avg_time > 0 else 0
        
        results.append({
            'Max Zone Time Limit': time_limit,
            'Fleet Size Required': fleet_size,
            'Commuter Ride Time (min)': est_commuter_ride_time
        })

    results_df = pd.DataFrame(results)
    
    # ==========================================
    # GENERATING THE GRAPH
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.plot(results_df['Commuter Ride Time (min)'], results_df['Fleet Size Required'], 
            marker='o', color='#8e44ad', linewidth=3, markersize=10)
    
    for i, txt in enumerate(results_df['Max Zone Time Limit']):
        ax.annotate(f"Max {txt}m", 
                    (results_df['Commuter Ride Time (min)'].iloc[i], results_df['Fleet Size Required'].iloc[i]), 
                    textcoords="offset points", xytext=(15,5), ha='left', fontsize=11, fontweight='bold')
                         
    # Assuming 25.0 is the baseline we established
    try:
        baseline_x = results_df[results_df['Max Zone Time Limit'] == 25.0]['Commuter Ride Time (min)'].values[0]
        ax.axvline(x=baseline_x, color='#27ae60', linestyle='--', linewidth=2, label='Chosen Pareto Point (25 Min Limit)')
    except IndexError:
        pass

    ax.set_title('Pareto Optimization: Commuter Travel Time vs. Operator Fleet Cost', fontsize=14, fontweight='bold')
    ax.set_xlabel('Commuter Cost: Estimated Ride Time (Minutes)', fontsize=12)
    ax.set_ylabel('Operator Cost: Fleet Size Required', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    output_img = "standalone_pareto_graph.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[SUCCESS] Generated Pareto graph to {output_img} without modifying main codebase.")

if __name__ == "__main__":
    run_standalone_analysis()