import pandas as pd
import folium

def isolate_commuters_by_rectangle(stops_file, station_name):
    print(f"Loading master stops data from {stops_file}...")
    try:
        df = pd.read_csv(stops_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{stops_file}'.")
        return

    # 1. Define a tight rectangle exactly around Andheri Station & Bus Depots
    MIN_LAT = 19.1140  
    MAX_LAT = 19.1240  
    MIN_LON = 72.8400  # Pushed further West to catch Andheri West station exits
    MAX_LON = 72.8520
    
    # 2. Find transit stops strictly inside this rectangle
    transit_stops = df[
        (df['stop_lat'] >= MIN_LAT) & (df['stop_lat'] <= MAX_LAT) &
        (df['stop_lon'] >= MIN_LON) & (df['stop_lon'] <= MAX_LON) &
        (df['likely_purpose'] == 'Other / Transit')
    ]
    
    target_commuters = transit_stops['device_aid'].unique()
    print(f"\nFound {len(target_commuters):,} true train commuters using the Rectangular Geofence!")

    if len(target_commuters) == 0:
        print("No commuters found. Adjust your bounding box coordinates.")
        return

    # 3. Extract complete itineraries and save
    commuter_df = df[df['device_aid'].isin(target_commuters)].copy()
    output_csv = f"{station_name.lower().replace(' ', '_')}_strict_commuters.csv"
    commuter_df.to_csv(output_csv, index=False)
    print(f"Saved complete itineraries to: {output_csv}")

    # ==========================================
    # VISUALIZING THE CATCHMENT & DESTINATIONS
    # ==========================================
    print("\nGenerating Catchment Area Map...")
    
    # Center map on Andheri Station
    center_lat = (MIN_LAT + MAX_LAT) / 2
    center_lon = (MIN_LON + MAX_LON) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

    # Draw the strict Rectangular Catchment Zone
    folium.Rectangle(
        bounds=[[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]],
        color='red',
        fill=True,
        fill_opacity=0.15,
        tooltip=f"{station_name} Catchment Box"
    ).add_to(m)

    # Plot destinations (Home/Work) safely
    destinations_df = commuter_df[commuter_df['likely_purpose'] != 'Other / Transit']
    
    # Safe sampling to prevent the ValueError
    safe_sample_size = min(1000, len(destinations_df))
    if safe_sample_size > 0:
        sample_df = destinations_df.sample(safe_sample_size)

        for _, row in sample_df.iterrows():
            color = 'blue' if 'Home' in row['likely_purpose'] else 'green'
            folium.CircleMarker(
                location=[row['stop_lat'], row['stop_lon']],
                radius=3, color=color, fill=True, fill_opacity=0.7,
                tooltip=row['likely_purpose']
            ).add_to(m)

    output_map = f"{station_name.lower().replace(' ', '_')}_strict_catchment_map.html"
    m.save(output_map)
    print(f"Success! Map saved as: {output_map}")

# Run Phase 1
isolate_commuters_by_rectangle('../mumbai_multiday_stops_robust.csv', 'Andheri Station')