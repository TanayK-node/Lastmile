import pandas as pd
import folium
import os

def generate_guaranteed_trajectory(feeder_csv, dispersal_csv):
    print("Hunting for a complete First-Mile and Last-Mile Trip Chain...")

    try:
        df_feeder = pd.read_csv(feeder_csv)
        df_dispersal = pd.read_csv(dispersal_csv)
    except FileNotFoundError:
        print("Error: Could not find the demand CSVs.")
        return

    # 1. The Fix: Merge ONLY on device_aid to find consistent commuters
    # This ignores missing daily data and links their known Home to their known Office
    merged = pd.merge(
        df_feeder, 
        df_dispersal, 
        on='device_aid', 
        suffixes=('_origin', '_dest')
    )

    if merged.empty:
        print("CRITICAL ERROR: No devices exist in both datasets.")
        print("Forcing a conceptual visualization using the top feeder and top dispersal...")
        # Fallback to force the visual if your data slice is too small
        sample_trip = {
            'device_aid': 'Simulated_User_Tanay_2026',
            'lat_origin': df_feeder.iloc[0]['lat'], 'lon_origin': df_feeder.iloc[0]['lon'],
            'station_origin': df_feeder.iloc[0]['station'],
            'lat_dest': df_dispersal.iloc[0]['lat'], 'lon_dest': df_dispersal.iloc[0]['lon'],
            'station_dest': df_dispersal.iloc[0]['station']
        }
    else:
        # Filter out trips where they boarded and alighted at the exact same station
        valid_trips = merged[merged['station_origin'] != merged['station_dest']]
        
        if not valid_trips.empty:
            sample_trip = valid_trips.iloc[0]
            print(f"SUCCESS! Found complete trip chain for Device: {sample_trip['device_aid']}")
        else:
            sample_trip = merged.iloc[0]
            print(f"Found match, but station is the same. Device: {sample_trip['device_aid']}")

    # 2. Coordinates Mapping
    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }

    m = folium.Map(location=[19.1197, 72.8464], zoom_start=12, tiles="CartoDB positron")

    # Extract Coordinates safely
    try:
        home_coords = [sample_trip['lat_origin'], sample_trip['lon_origin']]
        office_coords = [sample_trip['lat_dest'], sample_trip['lon_dest']]
        board_station = STATIONS[sample_trip['station_origin']]
        alight_station = STATIONS[sample_trip['station_dest']]
    except KeyError as e:
        print(f"Error mapping station coordinates. Make sure {e} is in the STATIONS dictionary.")
        return

    # 3. Draw Markers
    folium.Marker(home_coords, popup="<b>Home (Origin)</b>", icon=folium.Icon(color="blue", icon="home")).add_to(m)
    folium.Marker(board_station, popup=f"<b>Boarding: {sample_trip['station_origin']}</b>", icon=folium.Icon(color="black", icon="train", prefix="fa")).add_to(m)
    folium.Marker(alight_station, popup=f"<b>Alighting: {sample_trip['station_dest']}</b>", icon=folium.Icon(color="black", icon="train", prefix="fa")).add_to(m)
    folium.Marker(office_coords, popup="<b>Office (Destination)</b>", icon=folium.Icon(color="red", icon="briefcase")).add_to(m)

    # 4. Draw Trajectory Lines
    # First-Mile Line (Blue)
    folium.PolyLine(
        locations=[home_coords, board_station],
        color="blue", weight=4, opacity=0.8,
        tooltip="First-Mile (Feeder Leg)"
    ).add_to(m)

    # Train Ride Line (Dashed Gray)
    folium.PolyLine(
        locations=[board_station, alight_station],
        color="gray", weight=3, opacity=0.8, dash_array="10",
        tooltip="Suburban Train Journey"
    ).add_to(m)

    # Last-Mile Line (Red)
    folium.PolyLine(
        locations=[alight_station, office_coords],
        color="red", weight=4, opacity=0.8,
        tooltip="Last-Mile (Dispersal Leg)"
    ).add_to(m)

    m.fit_bounds([home_coords, office_coords, board_station, alight_station])

    output_filename = "guaranteed_trajectory_map_2026.html"
    m.save(output_filename)
    print(f"\n[+] Success! Trip chain mapped and saved to '{output_filename}'")

if __name__ == "__main__":
    FEEDER_FILE = "./demand_matrices/first_mile_feeder_demand.csv"
    DISPERSAL_FILE = "./demand_matrices/last_mile_dispersal_demand.csv"
    
    generate_guaranteed_trajectory(FEEDER_FILE, DISPERSAL_FILE)