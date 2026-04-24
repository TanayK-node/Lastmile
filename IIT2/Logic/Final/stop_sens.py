import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import os

# Import your pipeline modules
import osmnx_router as oxr
import route_clusterer as rc
import tsp_sequencer as tsp

def generate_virtual_stops(demand_df):
    if len(demand_df) < 5: return None
    coords = np.radians(demand_df[['lat', 'lon']])
    epsilon = (300 / 1000.0) / 6371.0088 
    db = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
    demand_df = demand_df.copy()
    demand_df['hub_cluster_id'] = db.labels_
    valid_clusters = demand_df[demand_df['hub_cluster_id'] != -1]
    
    virtual_stops = []
    stop_idx = 1
    for hub_id, hub_data in valid_clusters.groupby('hub_cluster_id'):
        virtual_stops.append({
            'matrix_idx': stop_idx, 'stop_id': f"Hub_{hub_id}",
            'lat': hub_data['lat'].mean(), 'lon': hub_data['lon'].mean(),
            'unique_commuters': hub_data['device_aid'].nunique()
        })
        stop_idx += 1

    if not virtual_stops:
        return None

    return pd.DataFrame(virtual_stops).sort_values(by='unique_commuters', ascending=False)

def run_city_sensitivity():
    print("==========================================")
    print("INITIATING CITY-WIDE SENSITIVITY ANALYSIS")
    print("==========================================\n")
    
    STATIONS = {
        "Andheri": (19.1197, 72.8464), "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573), "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }
    
    try:
        dfs = {
            "AM_F": pd.read_csv("./demand_matrices/AM_feeder_demand.csv"),
            "AM_D": pd.read_csv("./demand_matrices/AM_dispersal_demand.csv"),
            "PM_F": pd.read_csv("./demand_matrices/PM_feeder_demand.csv"),
            "PM_D": pd.read_csv("./demand_matrices/PM_dispersal_demand.csv")
        }
    except FileNotFoundError:
        print("Error: Demand CSVs not found.")
        return

    # Values to test (You suspected 3 is optimal, let's prove it)
    scenarios_to_test = [2, 3, 4, 5, 6, 8]
    results = []
    
    # We use smaller populations for the sensitivity sweep to keep it fast
    sweep_pop = 30
    sweep_gen = 30 
    
    for max_k in scenarios_to_test:
        print(f"\n--- Testing Constraint: Max {max_k} Stops per Route ---")
        
        city_times = []
        city_distances = []
        
        for station_name, (s_lat, s_lon) in STATIONS.items():
            # Dynamically load the station's road network
            G = oxr.load_or_download_station_graph(station_name, s_lat, s_lon)
            
            # CRITICAL FIX INCLUDED: Ensure we drop disconnected Infinity roads
            G = oxr.ox.truncate.largest_component(G, strongly=True)
            
            for _, scenario_df in dfs.items():
                station_demand = scenario_df[scenario_df['station'] == station_name]
                virtual_stops = generate_virtual_stops(station_demand)
                
                if virtual_stops is not None and len(virtual_stops) > 0:
                    dist_matrix, time_matrix, _ = oxr.build_osmnx_distance_matrix(G, s_lat, s_lon, virtual_stops.to_dict('records'))
                    
                    # Stage 1 GA with the current max_k being tested
                    zoned_stops = rc.cluster_hubs_into_routes(virtual_stops, time_matrix, max_stops_per_route=max_k, pop_size=sweep_pop, generations=sweep_gen)
                    
                    # Stage 2 TSP
                    for zone_id, zone_data in zoned_stops.groupby('route_zone_id'):
                        _, r_time, r_dist = tsp.optimize_cluster_route(zone_data.to_dict('records'), time_matrix, dist_matrix, depot_idx=0, pop_size=sweep_pop, generations=sweep_gen)
                        city_times.append(r_time)
                        city_distances.append(r_dist)
                        
        # Calculate Macro Metrics for this specific 'k' scenario across the whole city
        fleet_size = len(city_times)
        avg_time = np.mean(city_times) if fleet_size > 0 else 0
        total_vkt = sum(city_distances) / 1000.0 if fleet_size > 0 else 0
        
        results.append({
            'Max Stops Limit': max_k,
            'Fleet Size Required': fleet_size,
            'Avg Route Time (min)': avg_time,
            'System VKT (km)': total_vkt
        })
        
        print(f"Result for k={max_k} -> Fleet: {fleet_size} buses | Avg Time: {avg_time:.1f} mins | VKT: {total_vkt:.1f} km")

    results_df = pd.DataFrame(results)
    
    # ==========================================
    # ACADEMIC PLOTTING: THE PARETO FRONT
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Operator Cost vs. Route Constraint
    axes[0].plot(results_df['Max Stops Limit'], results_df['Fleet Size Required'], marker='o', color='#e74c3c', linewidth=2, label='Minibus Fleet Required')
    axes[0].set_title('Impact of Stop Constraints on Operator Fleet', fontsize=14)
    axes[0].set_xlabel('Constraint ($k$): Max Stops per Route Zone')
    axes[0].set_ylabel('Fleet Size Required (Operator Cost)')
    axes[0].legend()
    
    # Plot 2: THE PARETO FRONT (Commuter Time vs Fleet Size)
    axes[1].plot(results_df['Avg Route Time (min)'], results_df['Fleet Size Required'], marker='o', color='#2ecc71', linewidth=2)
    
    # Annotate the Pareto points with their max_k constraint
    for i, txt in enumerate(results_df['Max Stops Limit']):
        axes[1].annotate(f"k={txt}", (results_df['Avg Route Time (min)'].iloc[i], results_df['Fleet Size Required'].iloc[i]), 
                         textcoords="offset points", xytext=(10,10), ha='center', fontsize=11, fontweight='bold')
                         
    axes[1].set_title('Pareto Optimization: Commuter Time vs Operator Cost', fontsize=14)
    axes[1].set_xlabel('Commuter Cost: Average Route Cycle Time (Minutes)')
    axes[1].set_ylabel('Operator Cost: Fleet Size Required (Buses)')
    
    plt.tight_layout()
    output_img = "citywide_pareto_ctsem.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nAnalysis Complete! Saved Pareto graph to {output_img}")
    print("\n" + results_df.to_markdown(index=False))

if __name__ == "__main__":
    run_city_sensitivity()