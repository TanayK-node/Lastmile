import folium

def generate_geofence_map():
    print("Generating Station Geofencing Map...")
    
    # Center the map on Mumbai
    mumbai_center = [19.0760, 72.8777]
    m = folium.Map(location=mumbai_center, zoom_start=11, tiles="CartoDB positron")

    # Coordinates extracted directly from chain_extractor.py
    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }

    # The 600m mathematical radius used in the pipeline
    GEOFENCE_RADIUS_METERS = 600 

    for name, coords in STATIONS.items():
        # 1. Plot the Central Station Marker
        folium.Marker(
            location=coords,
            popup=f"<b>{name} Station</b>",
            icon=folium.Icon(color="black", icon="train", prefix='fa')
        ).add_to(m)

        # 2. Draw the 600m Geofence Catchment Circle
        folium.Circle(
            location=coords,
            radius=GEOFENCE_RADIUS_METERS,
            color="red",
            weight=2,
            fill=True,
            fill_color="red",
            fill_opacity=0.15,
            tooltip=f"{name} Catchment ({GEOFENCE_RADIUS_METERS}m)"
        ).add_to(m)

    output_filename = "station_geofences_map.html"
    m.save(output_filename)
    print(f"[+] Success! Map saved to '{output_filename}'")
    print("Open this file in your browser to capture the screenshot for your paper.")

if __name__ == "__main__":
    generate_geofence_map()