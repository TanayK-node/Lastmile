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
    
    # Set up our 4 explicit time-binned buckets
    am_feeder, am_dispersal = [], []
    pm_feeder, pm_dispersal = [], []
    
    print(f"Tracing 4D Commuter Flows (Max Local Bus Radius: {max_bus_radius/1000}km)...")
    
    for (device_id, trip_date), day_data in df.groupby(['device_aid', 'trip_date']):
        day_data = day_data.reset_index(drop=True)
        station_indices = day_data[day_data['location_tag'] != 'Non-Station'].index
        
        for idx in station_indices:
            station_name = day_data.loc[idx, 'location_tag']
            s_lat, s_lon = day_data.loc[idx, 'station_coords']
            
            # --- 1. FEEDER DEMAND (Home/Office -> Station) ---
            past_pings = day_data.loc[:idx-1]
            valid_origins = past_pings[past_pings['location_tag'] == 'Non-Station']
            
            if not valid_origins.empty:
                origin = valid_origins.iloc[-1]
                dist_to_station = calculate_distance(origin[lat_col], origin[lon_col], s_lat, s_lon)
                ping_hour = origin[time_col].hour
                
                # Verify it's a LOCAL trip (Under 6km)
                if dist_to_station <= max_bus_radius and dist_to_station > 200: # 200m prevents clustering the station itself
                    trip_data = {
                        'device_aid': device_id, 'date': trip_date,
                        'station': station_name, 'type': 'Feeder_Demand',
                        'lat': origin[lat_col], 'lon': origin[lon_col], 'hour': ping_hour
                    }
                    
                    # Bin by Time of Day
                    if 6 <= ping_hour < 12:     # 6 AM to 11:59 AM
                        am_feeder.append(trip_data)
                    elif 16 <= ping_hour < 22:  # 4 PM to 9:59 PM
                        pm_feeder.append(trip_data)
                
            # --- 2. DISPERSAL DEMAND (Station -> Office/Home) ---
            future_pings = day_data.loc[idx+1:]
            valid_dests = future_pings[future_pings['location_tag'] == 'Non-Station']
            
            if not valid_dests.empty:
                dest = valid_dests.iloc[0]
                dist_to_station = calculate_distance(dest[lat_col], dest[lon_col], s_lat, s_lon)
                ping_hour = dest[time_col].hour
                
                # Verify it's a LOCAL trip (Under 6km)
                if dist_to_station <= max_bus_radius and dist_to_station > 200:
                    trip_data = {
                        'device_aid': device_id, 'date': trip_date,
                        'station': station_name, 'type': 'Dispersal_Demand',
                        'lat': dest[lat_col], 'lon': dest[lon_col], 'hour': ping_hour
                    }
                    
                    # Bin by Time of Day
                    if 6 <= ping_hour < 12:     # 6 AM to 11:59 AM
                        am_dispersal.append(trip_data)
                    elif 16 <= ping_hour < 22:  # 4 PM to 9:59 PM
                        pm_dispersal.append(trip_data)

    # ==========================================
    # 3. SAVE AND EXPORT
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(am_feeder).to_csv(os.path.join(output_dir, "AM_feeder_demand.csv"), index=False)
    pd.DataFrame(am_dispersal).to_csv(os.path.join(output_dir, "AM_dispersal_demand.csv"), index=False)
    pd.DataFrame(pm_feeder).to_csv(os.path.join(output_dir, "PM_feeder_demand.csv"), index=False)
    pd.DataFrame(pm_dispersal).to_csv(os.path.join(output_dir, "PM_dispersal_demand.csv"), index=False)
    
    print("\n" + "="*50)
    print("4D EXTRACTION COMPLETE")
    print("="*50)
    print(f"Identified {len(am_feeder):,} AM Feeder trips | {len(am_dispersal):,} AM Dispersal trips.")
    print(f"Identified {len(pm_feeder):,} PM Feeder trips | {len(pm_dispersal):,} PM Dispersal trips.")

# ==========================================
if __name__ == "__main__":
    STOPS_DATA_FILE = 'mumbai_multiday_stops_robust.csv' # UPDATE IF NEEDED
    OUTPUT_DIRECTORY = './demand_matrices/'
    
    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }
    
    extract_trip_chains(STOPS_DATA_FILE, STATIONS, OUTPUT_DIRECTORY)