import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN

def get_distance(lat_array, lon_array, target_lat, target_lon):
    """Calculates distance in meters to a target coordinate."""
    R = 6371000  
    phi1, phi2 = np.radians(lat_array), np.radians(target_lat)
    dphi, dlambda = np.radians(target_lat - lat_array), np.radians(target_lon - lon_array)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def generate_smart_virtual_stops(commuter_file, station_lat, station_lon):
    print(f"Loading data from {commuter_file}...")
    try:
        df = pd.read_csv(commuter_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{commuter_file}'. Run Phase 1 first.")
        return

    # Keep only destinations (Home/Work)
    destinations = df[df['likely_purpose'] != 'Other / Transit'].copy()
    
    # ==========================================
    # 1. THE EXCLUSION ZONES (Anti-Catchment)
    # ==========================================
    competing_stations = {
        "Jogeshwari Station": (19.1363, 72.8489),
        "Vile Parle Station": (19.1006, 72.8440),
        "Goregaon Station": (19.1645, 72.8495)
    }
    
    print(f"\nStarting destination points: {len(destinations):,}")
    for name, (lat, lon) in competing_stations.items():
        # Calculate distance to competing station
        dists = get_distance(destinations['stop_lat'].values, destinations['stop_lon'].values, lat, lon)
        # Keep ONLY points MORE than 600 meters away
        destinations = destinations[dists > 600]
        print(f"Dropped points too close to {name}. Remaining: {len(destinations):,}")
    # ==========================================
    # NEW FIX: MAXIMUM OPERATIONAL RADIUS
    # ==========================================
    # Calculate how far every remaining point is from Andheri Station itself
    dists_from_origin = get_distance(destinations['stop_lat'].values, destinations['stop_lon'].values, station_lat, station_lon)
    
    # Drop anything further than 6 kilometers (6000 meters)
    destinations = destinations[dists_from_origin <= 6000]
    print(f"Dropped distant commuter homes (e.g., Mira Bhayandar). True local destinations remaining: {len(destinations):,}")
    # ==========================================
    if len(destinations) == 0:
        print("No valid destinations left after applying exclusion zones.")
        return

    # ==========================================
    # 2. UNIFIED AI CLUSTERING (DBSCAN)
    # ==========================================
    print("\nRunning DBSCAN Clustering to find Demand Hubs...")
    coords = np.radians(destinations[['stop_lat', 'stop_lon']])
    epsilon = (300 / 1000.0) / 6371.0088  # 300 meters
    
    # Find clusters with at least 4 distinct visits
    db = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
    destinations['hub_cluster_id'] = db.labels_

    valid_clusters = destinations[destinations['hub_cluster_id'] != -1]
    
    virtual_stops = []
    for hub_id, hub_data in valid_clusters.groupby('hub_cluster_id'):
        virtual_stops.append({
            'stop_id': f"Hub_{hub_id}",
            'lat': hub_data['stop_lat'].mean(),
            'lon': hub_data['stop_lon'].mean(),
            'unique_commuters': hub_data['device_aid'].nunique(),
            'total_visits': len(hub_data)
        })

    stops_df = pd.DataFrame(virtual_stops).sort_values(by='unique_commuters', ascending=False)
    stops_df.to_csv("smart_virtual_stops.csv", index=False)
    print(f"Successfully generated {len(stops_df)} Smart Virtual Stops! Saved to smart_virtual_stops.csv")

    # ==========================================
    # 3. MAPPING THE INFRASTRUCTURE
    # ==========================================
    print("\nGenerating Smart Infrastructure Map...")
    m = folium.Map(location=[station_lat, station_lon], zoom_start=13, tiles='openstreetmap')

    # Draw the Origin (Andheri)
    folium.Marker(
        [station_lat, station_lon], 
        popup="<b>Andheri Station (Origin)</b>", 
        icon=folium.Icon(color='red', icon='train', prefix='fa')
    ).add_to(m)

    # Draw the Exclusion Zones (Grey Circles)
    for name, (lat, lon) in competing_stations.items():
        folium.Circle(
            [lat, lon], radius=600, color='gray', fill=True, fill_opacity=0.3, 
            tooltip=f"Exclusion Zone: {name}"
        ).add_to(m)

    # Plot the Unified Demand Hubs (Virtual Bus Stops)
    for _, row in stops_df.iterrows():
        # Scale circle size by number of unique commuters
        radius_size = max(6, min(20, row['unique_commuters'] * 1.5))
        
        popup_html = f"""
        <div style="width:180px; font-family:Arial;">
            <h4 style="margin-bottom:5px; color:#b83b5e;">{row['stop_id']}</h4>
            <b>Unique Commuters:</b> {row['unique_commuters']}<br>
            <b>Total Recorded Visits:</b> {row['total_visits']}
        </div>
        """

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius_size,
            color='#b83b5e',
            fill=True, fill_color='#f08a5d', fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Demand Hub (Commuters: {row['unique_commuters']})"
        ).add_to(m)

    m.save("smart_virtual_bus_stops_map.html")
    print("Success! Map saved as smart_virtual_bus_stops_map.html")

# Run Phase 2
COMMUTER_FILE = 'andheri_station_strict_commuters.csv'
# Andheri Station Center Coordinates
generate_smart_virtual_stops(COMMUTER_FILE, 19.1197, 72.8464)