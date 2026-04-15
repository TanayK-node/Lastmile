import pandas as pd
import numpy as np

def get_distance(lat_array, lon_array, target_lat, target_lon):
    R = 6371000  
    phi1, phi2 = np.radians(lat_array), np.radians(target_lat)
    dphi, dlambda = np.radians(target_lat - lat_array), np.radians(target_lon - lon_array)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def run_sensitivity_analysis(stops_file, station_lat, station_lon):
    print(f"Loading master stops data from {stops_file}...\n")
    df = pd.read_csv(stops_file)

    # Define 5 different bounding boxes to test (from ultra-strict to very loose)
    # The offsets are roughly in degrees. 0.001 degrees is ~111 meters.
    test_boxes = [
        {"name": "Ultra-Strict (Platforms Only)", "lat_offset": 0.001, "lon_offset": 0.001},
        {"name": "Strict (Station + Rickshaw Stand)", "lat_offset": 0.002, "lon_offset": 0.003},
        {"name": "Optimal (Includes Bus Depots & Skywalks)", "lat_offset": 0.004, "lon_offset": 0.006},
        {"name": "Loose (Includes nearby cafes/shops)", "lat_offset": 0.006, "lon_offset": 0.010},
        {"name": "Too Large (Bleeds into next station)", "lat_offset": 0.010, "lon_offset": 0.015}
    ]

    results = []

    for box in test_boxes:
        # Calculate the actual coordinates for this test box
        min_lat = station_lat - box["lat_offset"]
        max_lat = station_lat + box["lat_offset"]
        min_lon = station_lon - box["lon_offset"]
        max_lon = station_lon + box["lon_offset"]
        
        # 1. Catchment Extraction (Phase 1)
        transit_stops = df[
            (df['stop_lat'] >= min_lat) & (df['stop_lat'] <= max_lat) &
            (df['stop_lon'] >= min_lon) & (df['stop_lon'] <= max_lon) &
            (df['likely_purpose'] == 'Other / Transit')
        ]
        commuter_ids = transit_stops['device_aid'].unique()
        num_commuters = len(commuter_ids)
        
        # 2. Virtual Stop Generation (Phase 2 Simulation)
        num_virtual_stops = 0
        total_demand = 0
        
        if num_commuters > 0:
            commuter_df = df[df['device_aid'].isin(commuter_ids)].copy()
            destinations = commuter_df[commuter_df['likely_purpose'] != 'Other / Transit'].copy()
            
            # Apply 6km operational radius filter
            dists_from_origin = get_distance(destinations['stop_lat'].values, destinations['stop_lon'].values, station_lat, station_lon)
            destinations = destinations[dists_from_origin <= 6000]
            
            if len(destinations) > 0:
                # Apply DBSCAN (Radius: 300m, Min Commuters: 4)
                from sklearn.cluster import DBSCAN
                coords = np.radians(destinations[['stop_lat', 'stop_lon']])
                epsilon = (300 / 1000.0) / 6371.0088 
                db = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
                destinations['hub_cluster_id'] = db.labels_
                
                valid_clusters = destinations[destinations['hub_cluster_id'] != -1]
                num_virtual_stops = valid_clusters['hub_cluster_id'].nunique()
                total_demand = valid_clusters['device_aid'].nunique()

        # Log the metrics
        results.append({
            "Box Size": box["name"],
            "Commuters Captured": num_commuters,
            "Virtual Hubs Generated": num_virtual_stops,
            "Total Hub Demand": total_demand
        })

    # Print the Final Evaluation Table
    print("="*80)
    print("CATCHMENT SENSITIVITY ANALYSIS RESULTS")
    print("="*80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("="*80)
    print("\nHow to use this data:")
    print("Look for the 'Elbow Point' - where increasing the box size stops capturing dense Hubs")
    print("and just starts adding scattered noise.")

# Run it!
RAW_STOPS_FILE = '../mumbai_multiday_stops_robust.csv'
STATION_LAT = 19.1197
STATION_LON = 72.8464

run_sensitivity_analysis(RAW_STOPS_FILE, STATION_LAT, STATION_LON)