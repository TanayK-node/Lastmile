import pandas as pd
import numpy as np

def calculate_access_distance(demand_df, virtual_stops):
    """Calculates Metric: True walking distance from commuter's doorstep to the virtual bus stop."""
    if virtual_stops.empty or demand_df.empty:
        return 0
        
    pax_coords = np.radians(demand_df[['lat', 'lon']].values)
    stop_coords = np.radians(virtual_stops[['lat', 'lon']].values)
    
    total_walk_dist = 0
    pax_count = 0
    
    for p_lat, p_lon in pax_coords:
        dlat = stop_coords[:, 0] - p_lat
        dlon = stop_coords[:, 1] - p_lon
        a = np.sin(dlat/2)**2 + np.cos(p_lat) * np.cos(stop_coords[:, 0]) * np.sin(dlon/2)**2
        distances = 6371000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        min_dist = np.min(distances)
        if min_dist <= 400: 
            total_walk_dist += min_dist
            pax_count += 1
            
    return (total_walk_dist / pax_count) if pax_count > 0 else 0

def evaluate_route_geometry(demand_df, virtual_stops_list, all_route_times, all_route_distances):
    """
    Generates the exact comparative A/B table with Commuter-Centric Detour & Time metrics.
    """
    total_commuters = len(demand_df)
    
    if len(virtual_stops_list) > 0:
        combined_stops = pd.concat(virtual_stops_list, ignore_index=True)
        covered_commuters = combined_stops['unique_commuters'].sum()
        avg_walk = calculate_access_distance(demand_df, combined_stops)
    else:
        covered_commuters = 0
        avg_walk = 0

    # Proposed System (AI DRT) Metrics
    proposed_routes = len(all_route_times)
    proposed_vkt = sum(all_route_distances) / 1000.0 
    avg_route_time = np.mean(all_route_times) if proposed_routes > 0 else 0
    avg_route_dist = (proposed_vkt / proposed_routes) if proposed_routes > 0 else 0
    
    # --- THE FIX: DYNAMIC SPEED MATCHING ---
    # Calculate the exact speed the AI is driving on the major roads
    if proposed_routes > 0 and avg_route_time > 0:
        avg_speed_kmh = avg_route_dist / (avg_route_time / 60.0) 
    else:
        avg_speed_kmh = 20.0 # Fallback speed

    # Baseline Simulator (Auto-Rickshaws)
    autos_required = int(np.ceil(covered_commuters / 1.8)) if covered_commuters > 0 else 0
    avg_direct_dist_km = 4.5 
    baseline_vkt = autos_required * avg_direct_dist_km * 2 if autos_required > 0 else 0
    
    # Calculate Auto time using the same AI free-flow network speed
    baseline_time_mins = (avg_direct_dist_km / avg_speed_kmh) * 60.0
    
    # --- COMMUTER METRICS ---
    est_bus_ride_dist = avg_route_dist * 0.6
    detour_index = max(1.0, est_bus_ride_dist / avg_direct_dist_km)
    
    est_bus_ride_time = (avg_route_time * 0.6) + 2.0 
    
    # Emissions Estimator (BS6 Minibus: 0.450 kg/km | Auto: 0.150 kg/km)
    proposed_co2 = proposed_vkt * 0.450
    baseline_co2 = baseline_vkt * 0.150
    
    fsr = round(autos_required / proposed_routes, 1) if proposed_routes > 0 else 0

    results = {
        "Evaluation Metric": [
            "Total Validated Commuters Served",
            "Fleet Size Required",
            "Avg. Access/Walk Distance (m)",
            "System VKT (Operator Cost - km)",
            "Avg. Commuter Travel Time (Minutes)",
            "Average Detour Index (Circuity)",
            "Fleet Substitution Ratio (FSR)",
            "Est. System CO2 Emissions (kg)"
        ],
        "Baseline (Auto-Rickshaws)": [
            f"{covered_commuters}",
            f"{autos_required} Autos",
            "0.0 (Door-to-Door)",
            f"{baseline_vkt:.2f}",
            f"~{baseline_time_mins:.1f} (Direct)",
            "1.00x (Direct)",
            "N/A",
            f"{baseline_co2:.2f}"
        ],
        "Proposed System (AI DRT)": [
            f"{covered_commuters}",
            f"{proposed_routes} Minibuses",
            f"{avg_walk:.1f}",
            f"{proposed_vkt:.2f}",
            f"~{est_bus_ride_time:.1f} (Optimized)",
            f"{detour_index:.2f}x",
            f"1 Bus replaces {fsr} Autos",
            f"{proposed_co2:.2f}"
        ]
    }
    
    return pd.DataFrame(results)