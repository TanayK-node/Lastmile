import pandas as pd
import numpy as np
import os
import glob

def haversine_distance_numpy(lat1, lon1, lat2, lon2):
    """Calculates the distance in meters between two GPS coordinates using NumPy arrays."""
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def extract_robust_stay_points(input_folder, output_stops_file, 
                               dist_threshold_m=150, time_threshold_mins=15):
    """
    Now uses literature-backed thresholds (150m) and calculates the exact 
    number of GPS pings that make up each stop (Confidence Score).
    """
    print(f"Scanning '{input_folder}' and all subfolders...")
    search_pattern = os.path.join(input_folder, "**", "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    if not csv_files:
        print("No CSV files found.")
        return
        
    all_data = [pd.read_csv(f) for f in csv_files]
    master_df = pd.concat(all_data, ignore_index=True)
    master_df = master_df.sort_values(by=['device_aid', 'timestamp']).reset_index(drop=True)
    master_df['datetime_ist'] = pd.to_datetime(master_df['timestamp'], unit='s') + pd.Timedelta(hours=5, minutes=30)
    
    stops_list = []
    grouped = master_df.groupby('device_aid')
    print(f"Analyzing {len(grouped):,} unique devices...\n")

    processed_count = 0

    for device, device_df in grouped:
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} / {len(grouped)} devices...")

        if len(device_df) < 2:
            continue
            
        lats = device_df['latitude'].values
        lons = device_df['longitude'].values
        times = device_df['timestamp'].values 
        datetimes = device_df['datetime_ist'].values
        
        n_points = len(lats)
        i = 0
        
        while i < n_points - 1:
            start_lat, start_lon, start_time = lats[i], lons[i], times[i]
            j = i + 1
            
            while j < n_points:
                dist = haversine_distance_numpy(start_lat, start_lon, lats[j], lons[j])
                if dist > dist_threshold_m:
                    break
                j += 1
                
            j = j - 1
            time_diff_mins = (times[j] - start_time) / 60.0
            
            if time_diff_mins >= time_threshold_mins:
                # NEW: Calculate the exact number of pings captured during this stop
                ping_count = (j - i) + 1
                
                stop_lat = np.mean(lats[i:j+1])
                stop_lon = np.mean(lons[i:j+1])
                
                stops_list.append({
                    'device_aid': device,
                    'arrival_time': datetimes[i],
                    'departure_time': datetimes[j],
                    'duration_mins': round(time_diff_mins, 1),
                    'ping_count': ping_count,        # <--- Added Confidence Score
                    'stop_lat': stop_lat,
                    'stop_lon': stop_lon
                })
                i = j + 1 
            else:
                i += 1

    print("\n" + "="*40)
    print("STAY POINT EXTRACTION COMPLETE")
    print("="*40)
    
    stops_df = pd.DataFrame(stops_list)
    
    if len(stops_df) > 0:
        print(f"Identified {len(stops_df):,} distinct stops.")
        
        stops_df['arrival_time'] = pd.to_datetime(stops_df['arrival_time'])
        stops_df['hour_of_day'] = stops_df['arrival_time'].dt.hour
        
        stops_df['likely_purpose'] = np.where(
            (stops_df['hour_of_day'] >= 21) | (stops_df['hour_of_day'] <= 6), 'Home (Overnight)', 
            np.where((stops_df['hour_of_day'] >= 8) & (stops_df['hour_of_day'] <= 18) & (stops_df['duration_mins'] > 180), 'Work/School', 'Other / Transit')
        )
        
        # Optional: Filter out "ghost stops" with very low ping counts
        stops_df = stops_df[stops_df['ping_count'] >= 3]
        
        stops_df.to_csv(output_stops_file, index=False)
        print(f"Saved highly structured stop data to: {output_stops_file}")
    else:
        print("No stops detected.")

# ==========================================
# HOW TO USE
# ==========================================
INPUT_FOLDER = '../mumbai_active_data' 
OUTPUT_STOPS_FILE = 'mumbai_multiday_stops_robust.csv'

# Now using 150 meters and 15 minutes as our baseline
extract_robust_stay_points(INPUT_FOLDER, OUTPUT_STOPS_FILE, dist_threshold_m=150, time_threshold_mins=15)