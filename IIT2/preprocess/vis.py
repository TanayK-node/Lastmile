import pandas as pd
import folium
import glob
import os

def map_full_journey(input_folder, specific_device_id=None):
    """
    Combines all cleaned CSVs in a folder, tracks a specific device, 
    and maps the journey using different colors for each source file.
    """
    print(f"Scanning folder: '{input_folder}' for CSV files...")
    
    # 1. Find all CSV files in the folder
    search_pattern = os.path.join(input_folder, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print("No CSV files found! Check your folder path.")
        return

    # A list of distinct colors supported by Folium
    color_palette = [
        'blue', 'purple', 'orange', 'darkred', 'cadetblue', 
        'darkgreen', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 
        'gray', 'black', 'lightgray', 'red', 'green'
    ]
    
    # 2. Load and Combine all Data
    all_data_frames = []
    file_color_mapping = {} # To keep track of which file gets which color
    
    for index, file_path in enumerate(csv_files):
        filename = os.path.basename(file_path)
        
        # Pick a color from the palette (loop back to start if we have more than 15 files)
        assigned_color = color_palette[index % len(color_palette)]
        file_color_mapping[filename] = assigned_color
        
        print(f"Loading {filename} -> Assigned Color: {assigned_color}")
        
        df = pd.read_csv(file_path)
        
        # Add tracking columns before merging
        df['source_file'] = filename
        df['point_color'] = assigned_color
        
        # Convert timestamp for tooltips
        if 'timestamp' in df.columns:
            df['readable_time'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        all_data_frames.append(df)

    # Stitch them all together into one massive dataset
    master_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"\nSuccessfully combined {len(csv_files)} files. Total rows: {len(master_df):,}")

    # 3. Decide which device to track
    if specific_device_id is None:
        specific_device_id = master_df['device_aid'].value_counts().index[0]
        print(f"Auto-selected the most active device overall: {specific_device_id}")
    else:
        if specific_device_id not in master_df['device_aid'].values:
            print(f"Error: Device {specific_device_id} not found in any of the files.")
            return

    # 4. Filter and sort the master dataset
    device_data = master_df[master_df['device_aid'] == specific_device_id].copy()
    device_data = device_data.sort_values('timestamp') # Crucial: Sort by time across ALL files
    
    coords = device_data[['latitude', 'longitude']].values.tolist()
    
    if len(coords) < 2:
        print("Not enough points to draw a path.")
        return

    print(f"Plotting {len(coords)} total points for device {specific_device_id}...")

    # 5. Create the Map
    center_lat = device_data['latitude'].mean()
    center_lon = device_data['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')

    # Draw the main connecting path as a neutral dark gray line
    folium.PolyLine(
        locations=coords,
        color='darkgray',
        weight=2,
        opacity=0.7,
        tooltip="Overall Travel Path"
    ).add_to(m)

    # 6. Plot every single point with its file-specific color
    for _, row in device_data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=row['point_color'],
            fill=True,
            fill_color=row['point_color'],
            fill_opacity=0.9,
            tooltip=f"<b>Time:</b> {row['readable_time']}<br><b>File:</b> {row['source_file']}"
        ).add_to(m)

    # Highlight the very first point so the journey start is easy to spot.
    folium.CircleMarker(
        location=coords[0],
        radius=9,
        color='green',
        fill=True,
        fill_color='yellow',
        fill_opacity=1.0,
        weight=3,
        tooltip="Journey Start Point"
    ).add_to(m)

    # 7. Add Start and End Markers
    folium.Marker(
        location=coords[0], popup="Journey Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    folium.Marker(
        location=coords[-1], popup="Journey End",
        icon=folium.Icon(color="black", icon="stop")
    ).add_to(m)

    # 8. Add a custom Legend to the Map
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 300px; max-height: 400px; overflow-y: auto;
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px; border-radius: 5px;">
     <b>Data Source Legend</b><br>
     <hr style="margin: 5px 0;">
     '''
    for filename, color in file_color_mapping.items():
        legend_html += f'<i style="background:{color}; width:12px; height:12px; float:left; margin-right:8px; border-radius:50%;"></i>{filename}<br>'
    legend_html += '<hr style="margin: 8px 0;">'
    legend_html += '<i style="background:yellow; border:3px solid green; width:12px; height:12px; float:left; margin-right:8px; border-radius:50%;"></i>Journey Start Point<br>'
    legend_html += '<i style="background:black; width:12px; height:12px; float:left; margin-right:8px; border-radius:50%;"></i>Journey End Point<br>'
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))

    # 9. Save the Map
    output_filename = f"full_journey_{specific_device_id}.html"
    m.save(output_filename)
    
    print(f"\nSuccess! Full multi-file map saved as: {output_filename}")

# ==========================================
# HOW TO USE
# ==========================================
# 1. Point this to the folder containing your formatted CSVs
CLEANED_FOLDER = '../mumbai_data/01_mumbai'

# 2. Paste the specific device ID you want to track, or leave as None to auto-find the busiest
TARGET_DEVICE = "29c0c9b7-3b0b-4f7e-a86a-004a7a148857"

# Run it!
map_full_journey(CLEANED_FOLDER, TARGET_DEVICE)