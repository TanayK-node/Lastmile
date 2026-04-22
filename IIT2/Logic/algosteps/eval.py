import pandas as pd
import numpy as np
import os

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula to calculate distance in meters."""
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def evaluate_spatial_metrics(feeder_csv, dispersal_csv, stations_dict, output_csv):
    print("==========================================")
    print("CALCULATING SPATIAL ACCESSIBILITY METRICS")
    print("==========================================\n")
    
    results = []

    # 1. Process First-Mile (Feeder)
    if os.path.exists(feeder_csv):
        df_feeder = pd.read_csv(feeder_csv)
        
        # Calculate distance from home to station for every commuter
        distances = []
        for _, row in df_feeder.iterrows():
            s_lat, s_lon = stations_dict[row['station']]
            dist = calculate_distance(row['lat'], row['lon'], s_lat, s_lon)
            distances.append(dist)
            
        df_feeder['access_distance_m'] = distances
        
        for station_name, group in df_feeder.groupby('station'):
            results.append({
                'Station': station_name,
                'Direction': 'First-Mile (Feeder)',
                'Total_Commuters': len(group),
                'Avg_Access_Distance_km': round(group['access_distance_m'].mean() / 1000, 2),
                'Max_Catchment_Radius_km': round(group['access_distance_m'].max() / 1000, 2)
            })

    # 2. Process Last-Mile (Dispersal)
    if os.path.exists(dispersal_csv):
        df_dispersal = pd.read_csv(dispersal_csv)
        
        # Calculate distance from station to office for every commuter
        distances = []
        for _, row in df_dispersal.iterrows():
            s_lat, s_lon = stations_dict[row['station']]
            dist = calculate_distance(row['lat'], row['lon'], s_lat, s_lon)
            distances.append(dist)
            
        df_dispersal['egress_distance_m'] = distances
        
        for station_name, group in df_dispersal.groupby('station'):
            results.append({
                'Station': station_name,
                'Direction': 'Last-Mile (Dispersal)',
                'Total_Commuters': len(group),
                'Avg_Access_Distance_km': round(group['egress_distance_m'].mean() / 1000, 2),
                'Max_Catchment_Radius_km': round(group['egress_distance_m'].max() / 1000, 2)
            })

    # 3. Format and Export
    results_df = pd.DataFrame(results)
    
    # Sort for clean reading
    results_df = results_df.sort_values(by=['Station', 'Direction']).reset_index(drop=True)
    
    print(results_df.to_markdown(index=False))
    
    results_df.to_csv(output_csv, index=False)
    print(f"\n[+] Success! Evaluation metrics saved to {output_csv}")

if __name__ == "__main__":
    FEEDER_FILE = "./demand_matrices/first_mile_feeder_demand.csv"
    DISPERSAL_FILE = "./demand_matrices/last_mile_dispersal_demand.csv"
    OUTPUT_METRICS = "academic_evaluation_metrics.csv"
    
    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }
    
    evaluate_spatial_metrics(FEEDER_FILE, DISPERSAL_FILE, STATIONS, OUTPUT_METRICS)