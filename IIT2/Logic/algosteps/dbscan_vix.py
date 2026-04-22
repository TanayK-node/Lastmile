import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
import os

def generate_citywide_cluster_map(demand_csv):
    print("==========================================")
    print("INITIALIZING CITY-WIDE DBSCAN CLUSTERING")
    print("==========================================\n")
    print(f"Loading real demand data from {demand_csv}...")

    try:
        df = pd.read_csv(demand_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {demand_csv}. Make sure you ran the extraction step.")
        return

    # Center the map on Mumbai
    m = folium.Map(location=[19.1000, 72.8400], zoom_start=11, tiles="CartoDB positron")

    # Define the 5 stations with distinct color identities
    STATIONS = {
        "Andheri": {"coords": (19.1197, 72.8464), "color": "blue", "light_color": "lightblue"},
        "Bandra": {"coords": (19.0544, 72.8402), "color": "red", "light_color": "lightcoral"},
        "Borivali": {"coords": (19.2291, 72.8573), "color": "green", "light_color": "lightgreen"},
        "Goregaon": {"coords": (19.1645, 72.8495), "color": "purple", "light_color": "#DDA0DD"}, # Plum
        "Churchgate": {"coords": (18.9322, 72.8264), "color": "orange", "light_color": "#FFD580"} # Light Orange
    }

    # The mathematical radius for DBSCAN (300 meters)
    epsilon = (300 / 1000.0) / 6371.0088 

    for station_name, props in STATIONS.items():
        s_lat, s_lon = props["coords"]
        main_color = props["color"]
        light_color = props["light_color"]

        # 1. Plot the Central Transit Hub
        folium.Marker(
            [s_lat, s_lon], 
            popup=f"<b>{station_name} Station</b>", 
            icon=folium.Icon(color=main_color, icon='train', prefix='fa')
        ).add_to(m)

        # Filter real data for this specific station
        station_df = df[df['station'] == station_name].copy()
        
        if len(station_df) < 5:
            print(f"  [-] Skipping {station_name}: Insufficient demand points.")
            continue

        # 2. Run DBSCAN Clustering Algorithm
        coords = np.radians(station_df[['lat', 'lon']])
        db = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
        station_df['hub_cluster_id'] = db.labels_
        
        num_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        print(f"  [+] {station_name}: Generated {num_clusters} Virtual Stops.")

        # 3. Plot Raw Demand Points (Signal vs. Noise)
        for _, row in station_df.iterrows():
            is_noise = (row['hub_cluster_id'] == -1)
            # Noise is gray, Clustered points take the station's light color
            point_color = 'gray' if is_noise else light_color
            opacity = 0.3 if is_noise else 0.6
            radius = 2 if is_noise else 3
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                color=point_color,
                fill=True,
                fill_opacity=opacity,
                weight=0
            ).add_to(m)

        # 4. Plot the Virtual Bus Stops (Cluster Centers)
        valid_clusters = station_df[station_df['hub_cluster_id'] != -1]
        for hub_id, hub_data in valid_clusters.groupby('hub_cluster_id'):
            hub_lat = hub_data['lat'].mean()
            hub_lon = hub_data['lon'].mean()
            commuter_count = hub_data['device_aid'].nunique()
            
            folium.CircleMarker(
                location=[hub_lat, hub_lon],
                radius=7,
                color=main_color,
                fill=True,
                fill_opacity=0.9,
                weight=2,
                tooltip=f"<b>{station_name} - Hub {hub_id}</b><br>Demand: {commuter_count} pax"
            ).add_to(m)

    output_filename = "citywide_dbscan_clusters.html"
    m.save(output_filename)
    print(f"\n[+] Success! City-wide clustering map saved to '{output_filename}'")
    print("Open this file in your browser to view all 5 clustered networks simultaneously.")

if __name__ == "__main__":
    # We will use the First-Mile Feeder demand as it nicely illustrates residential neighborhood clustering
    FEEDER_FILE = "./demand_matrices/first_mile_feeder_demand.csv"
    
    generate_citywide_cluster_map(FEEDER_FILE)