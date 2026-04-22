import pandas as pd
import numpy as np
import os

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula to calculate distance in meters."""
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def tag_stations(lat, lon, stations_dict, radius=600): # Increased to 600m to catch noisy GPS
    for station_name, coords in stations_dict.items():
        if calculate_distance(lat, lon, coords[0], coords[1]) <= radius:
            return station_name, coords
    return "Non-Station", None

# ==========================================
# 2. CORE EXTRACTION LOGIC
# ==========================================
def extract_trip_chains(stops_file_path, stations_dict, output_dir, max_bus_radius=6000):
    print(f"Loading Stay Points Data from: {stops_file_path}")
    
    try:
        df = pd.read_csv(stops_file_path)
    except FileNotFoundError:
        print("Error: Could not find the file.")
        return
        
    lat_col = 'lat' if 'lat' in df.columns else 'latitude' if 'latitude' in df.columns else 'stop_lat'
    lon_col = 'lon' if 'lon' in df.columns else 'longitude' if 'longitude' in df.columns else 'stop_lon'
    time_col = 'arrival_time' if 'arrival_time' in df.columns else 'timestamp'

    df[time_col] = pd.to_datetime(df[time_col])
    df['trip_date'] = df[time_col].dt.date
    df = df.sort_values(by=['device_aid', time_col]).reset_index(drop=True)
    
    print("Geofencing Station Catchments (600m radius)...")
    tags = df.apply(lambda row: tag_stations(row[lat_col], row[lon_col], stations_dict), axis=1)
    df['location_tag'] = [t[0] for t in tags]
    df['station_coords'] = [t[1] for t in tags]
    
    first_mile_demands = []
    last_mile_demands = []
    
    print(f"Tracing Commuters (Max Local Bus Radius: {max_bus_radius/1000}km)...")
    
    for (device_id, trip_date), day_data in df.groupby(['device_aid', 'trip_date']):
        day_data = day_data.reset_index(drop=True)
        station_indices = day_data[day_data['location_tag'] != 'Non-Station'].index
        
        for idx in station_indices:
            station_name = day_data.loc[idx, 'location_tag']
            s_lat, s_lon = day_data.loc[idx, 'station_coords']
            
            # --- 1. FIRST MILE (Home -> Station) ---
            past_pings = day_data.loc[:idx-1]
            valid_origins = past_pings[past_pings['location_tag'] == 'Non-Station']
            
            if not valid_origins.empty:
                home = valid_origins.iloc[-1]
                dist_to_station = calculate_distance(home[lat_col], home[lon_col], s_lat, s_lon)
                
                # Verify it's a LOCAL trip (Under 6km)
                if dist_to_station <= max_bus_radius and dist_to_station > 200: # 200m prevents clustering the station itself
                    first_mile_demands.append({
                        'device_aid': device_id, 'date': trip_date,
                        'station': station_name, 'type': 'Feeder_Demand',
                        'lat': home[lat_col], 'lon': home[lon_col]
                    })
                
            # --- 2. LAST MILE (Station -> Office) ---
            future_pings = day_data.loc[idx+1:]
            valid_dests = future_pings[future_pings['location_tag'] == 'Non-Station']
            
            if not valid_dests.empty:
                office = valid_dests.iloc[0]
                dist_to_station = calculate_distance(office[lat_col], office[lon_col], s_lat, s_lon)
                
                # Verify it's a LOCAL trip (Under 6km)
                if dist_to_station <= max_bus_radius and dist_to_station > 200:
                    last_mile_demands.append({
                        'device_aid': device_id, 'date': trip_date,
                        'station': station_name, 'type': 'Dispersal_Demand',
                        'lat': office[lat_col], 'lon': office[lon_col]
                    })

    # ==========================================
    # 3. SAVE AND EXPORT
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(first_mile_demands).to_csv(os.path.join(output_dir, "first_mile_feeder_demand.csv"), index=False)
    pd.DataFrame(last_mile_demands).to_csv(os.path.join(output_dir, "last_mile_dispersal_demand.csv"), index=False)
    
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"Identified {len(first_mile_demands):,} First-Mile Feeder trips.")
    print(f"Identified {len(last_mile_demands):,} Last-Mile Dispersal trips.")

# ==========================================
if __name__ == "__main__":
    STOPS_DATA_FILE = '../mumbai_multiday_stops_robust.csv' # UPDATE IF NEEDED
    OUTPUT_DIRECTORY = './demand_matrices/'
    
    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }
    
    extract_trip_chains(STOPS_DATA_FILE, STATIONS, OUTPUT_DIRECTORY)