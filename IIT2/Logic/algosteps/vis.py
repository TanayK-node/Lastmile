import pandas as pd
import folium
import os

def generate_demand_map(feeder_csv, dispersal_csv, output_html):
    print("Initializing Multi-Station Demand Map...")
    
    # Center map on Mumbai
    mumbai_center = [19.1197, 72.8464] 
    m = folium.Map(location=mumbai_center, zoom_start=11, tiles="CartoDB positron")

    # The 5 Core Stations
    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }

    # Feature Groups for toggling layers on/off
    fg_stations = folium.FeatureGroup(name="Transit Hubs (Stations)")
    fg_feeder = folium.FeatureGroup(name="First-Mile Demand (Home -> Station)", show=True)
    fg_dispersal = folium.FeatureGroup(name="Last-Mile Demand (Station -> Office)", show=True)

    # 1. Plot the Stations
    for name, coords in STATIONS.items():
        folium.Marker(
            location=coords,
            popup=f"<b>{name} Station</b>",
            icon=folium.Icon(color="black", icon="train", prefix='fa')
        ).add_to(fg_stations)

    # 2. Plot First-Mile (Feeder) - Blue
    if os.path.exists(feeder_csv):
        print("Plotting First-Mile Feeder data...")
        df_feeder = pd.read_csv(feeder_csv)
        for _, row in df_feeder.iterrows():
            station_coords = STATIONS.get(row['station'])
            if station_coords:
                # Draw the commuter's location
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=3, color="blue", fill=True, fill_opacity=0.6,
                    popup="First-Mile Origin"
                ).add_to(fg_feeder)
                
                # Draw a faint line to the station
                folium.PolyLine(
                    locations=[[row['lat'], row['lon']], station_coords],
                    color="blue", weight=0.5, opacity=0.3
                ).add_to(fg_feeder)

    # 3. Plot Last-Mile (Dispersal) - Red
    if os.path.exists(dispersal_csv):
        print("Plotting Last-Mile Dispersal data...")
        df_dispersal = pd.read_csv(dispersal_csv)
        for _, row in df_dispersal.iterrows():
            station_coords = STATIONS.get(row['station'])
            if station_coords:
                # Draw the commuter's destination
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=3, color="red", fill=True, fill_opacity=0.6,
                    popup="Last-Mile Destination"
                ).add_to(fg_dispersal)
                
                # Draw a faint line back to the station
                folium.PolyLine(
                    locations=[station_coords, [row['lat'], row['lon']]],
                    color="red", weight=0.5, opacity=0.3
                ).add_to(fg_dispersal)

    # Assemble Map
    m.add_child(fg_stations)
    m.add_child(fg_feeder)
    m.add_child(fg_dispersal)
    folium.LayerControl().add_to(m)

    m.save(output_html)
    print(f"\nSuccess! Map saved to: {output_html}")
    print("Open this file in your web browser to view the hub-and-spoke network.")

if __name__ == "__main__":
    FEEDER_FILE = "./demand_matrices/first_mile_feeder_demand.csv"
    DISPERSAL_FILE = "./demand_matrices/last_mile_dispersal_demand.csv"
    OUTPUT_MAP = "multi_station_demand_map.html"
    
    generate_demand_map(FEEDER_FILE, DISPERSAL_FILE, OUTPUT_MAP)