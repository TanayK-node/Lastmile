import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx

def calculate_access_distance(demand_df, virtual_stops):
    """
    Robustly calculates average walking distance by matching each commuter 
    to their nearest generated virtual stop, ignoring arbitrary cluster IDs.
    """
    if virtual_stops.empty or demand_df.empty:
        return 0
        
    # Convert all coordinates to radians for the Haversine formula
    pax_coords = np.radians(demand_df[['lat', 'lon']].values)
    stop_coords = np.radians(virtual_stops[['lat', 'lon']].values)
    
    total_walk_dist = 0
    pax_count = 0
    
    for p_lat, p_lon in pax_coords:
        # Calculate distance from this single passenger to ALL virtual stops simultaneously
        dlat = stop_coords[:, 0] - p_lat
        dlon = stop_coords[:, 1] - p_lon
        
        a = np.sin(dlat/2)**2 + np.cos(p_lat) * np.cos(stop_coords[:, 0]) * np.sin(dlon/2)**2
        distances = 6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        # Find the absolute closest virtual stop to this commuter
        min_dist = np.min(distances)
        
        # Since our DBSCAN epsilon was 300m, anyone further than ~400m was classified 
        # as "noise" (not taking the bus). We only average the valid commuters.
        if min_dist <= 400:
            total_walk_dist += min_dist
            pax_count += 1
            
    return (total_walk_dist / pax_count) if pax_count > 0 else 0


def evaluate_fleet_performance(G, routes, demand_df, virtual_stops, station_lat, station_lon, bus_capacity=10):
    total_pax = sum(sum(stop['load'] for stop in route) for route in routes)
    fleet_size = len(routes)
    
    # 1. Capacity Utilization
    max_system_capacity = fleet_size * bus_capacity
    utilization_rate = (total_pax / max_system_capacity) * 100 if max_system_capacity > 0 else 0
    
    # 2. Extract AI Route Distances
    ai_vkt = 0
    ai_pax_meters = 0
    
    for route in routes:
        current_load = 0
        last_node = ox.distance.nearest_nodes(G, X=station_lon, Y=station_lat)
        for stop in route:
            stop_node = ox.distance.nearest_nodes(G, X=stop['lon'], Y=stop['lat'])
            try:
                dist = nx.shortest_path_length(G, last_node, stop_node, weight='length')
            except nx.NetworkXNoPath:
                dist = 0
            ai_vkt += dist
            ai_pax_meters += (current_load * dist)
            current_load += stop['load']
            last_node = stop_node
            
        depot_node = ox.distance.nearest_nodes(G, X=station_lon, Y=station_lat)
        try:
            dist = nx.shortest_path_length(G, last_node, depot_node, weight='length')
        except nx.NetworkXNoPath:
            dist = 0
        ai_vkt += dist
        ai_pax_meters += (current_load * dist)

    ai_avg_ride_dist = ai_pax_meters / total_pax if total_pax > 0 else 0

    # 3. Baseline Simulation: Direct Auto-Rickshaws
    baseline_vkt = 0
    baseline_pax_meters = 0
    baseline_autos = int(sum(np.ceil(virtual_stops['unique_commuters'] / 2)))
    # Autos usually need an empty repositioning leg in addition to the loaded trip.
    # Using 2.0 keeps the vehicle-km accounting comparable with the minibus system,
    # where all deadhead movement is already counted in ai_vkt.
    AUTO_DEADHEAD_MULTIPLIER = 2.0
    
    depot_node = ox.distance.nearest_nodes(G, X=station_lon, Y=station_lat)
    for _, stop in virtual_stops.iterrows():
        stop_node = ox.distance.nearest_nodes(G, X=stop['lon'], Y=stop['lat'])
        try:
            direct_dist = nx.shortest_path_length(G, stop_node, depot_node, weight='length')
        except nx.NetworkXNoPath:
            direct_dist = 0
            
        load = stop['unique_commuters']
        autos_needed = np.ceil(load / 2)
        baseline_vkt += (autos_needed * direct_dist * AUTO_DEADHEAD_MULTIPLIER)
        baseline_pax_meters += (load * direct_dist)  
        
    baseline_avg_ride_dist = baseline_pax_meters / total_pax if total_pax > 0 else 0
    
    # 4. ACADEMIC METRICS FIX
    ai_avg_walk = calculate_access_distance(demand_df, virtual_stops)
    detour_index = ai_avg_ride_dist / baseline_avg_ride_dist if baseline_avg_ride_dist > 0 else 1
    fsr = baseline_autos / fleet_size if fleet_size > 0 else 0
    
    CO2_PER_KM_AUTO = 0.150
    CO2_PER_KM_MINIBUS = 0.450
    
    baseline_co2 = (baseline_vkt / 1000) * CO2_PER_KM_AUTO
    ai_co2 = (ai_vkt / 1000) * CO2_PER_KM_MINIBUS

    # Compile the Results
    results = {
        "Evaluation Metric": [
            "Total Validated Commuters Served",
            "Fleet Size Required",
            "Fleet Capacity Utilization (%)",
            "Avg. Access/Walk Distance (m)", 
            "System VKT (Operator Cost - km)",
            "Avg. Commuter Ride Distance (km)",
            "Average Detour Index (Circuity)",
            "Fleet Substitution Ratio (FSR)",
            "Est. System CO2 Emissions (kg)"
        ],
        "Baseline (Auto-Rickshaws)": [
            total_pax,
            f"{baseline_autos} Autos",
            "N/A",
            "0.0 (Door-to-Door)",
            f"{baseline_vkt / 1000:.2f}",
            f"{baseline_avg_ride_dist / 1000:.2f}",
            "1.00 (Direct)",
            "N/A",
            f"{baseline_co2:.2f}"
        ],
        "Proposed System (AI DRT)": [
            total_pax,
            f"{fleet_size} Minibuses",
            f"{utilization_rate:.1f}%",
            f"{ai_avg_walk:.1f}", 
            f"{ai_vkt / 1000:.2f}",
            f"{ai_avg_ride_dist / 1000:.2f}",
            f"{detour_index:.2f}x",
            f"1 Bus replaces {fsr:.1f} Autos",
            f"{ai_co2:.2f}"
        ]
    }
    
    return pd.DataFrame(results)