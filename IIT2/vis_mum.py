import pandas as pd
import folium
import glob
import os

def map_combined_daily_files(input_folder, specific_device_id=None):
    """
    Reads all daily highly-active CSV files from a folder, combines them,
    and maps the multiday vehicle patterns with color-coded dates.
    """
    print(f"Scanning '{input_folder}' for daily master files...")
    
    # 1. Find all CSV files in the folder
    search_pattern = os.path.join(input_folder, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return
        
    print(f"Found {len(csv_files)} daily files. Combining data...")
    
    # 2. Combine all daily files into one DataFrame
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")
            
    if not all_data:
        print("No valid data could be loaded.")
        return
        
    master_df = pd.concat(all_data, ignore_index=True)
    
    # 3. Process Timestamps and Dates
    master_df['datetime'] = pd.to_datetime(master_df['timestamp'], unit='s', errors='coerce')
    master_df['date'] = master_df['datetime'].dt.date
    master_df['time_only'] = master_df['datetime'].dt.strftime('%H:%M:%S')

    print(f"Successfully combined {len(master_df):,} total points across {master_df['date'].nunique()} days.")

    # 4. Select the Device
    if specific_device_id is None:
        specific_device_id = master_df['device_aid'].value_counts().index[0]
        print(f"Auto-selected the most active vehicle/device: {specific_device_id}")
    else:
        if specific_device_id not in master_df['device_aid'].values:
            print(f"Error: Device {specific_device_id} not found in the combined data.")
            return

    # 5. Filter and Sort Chronologically (Crucial for trip segmentation)
    device_data = master_df[master_df['device_aid'] == specific_device_id].copy()
    device_data = device_data.sort_values('datetime')
    
    unique_days = device_data['date'].nunique()
    print(f"\nPlotting {len(device_data):,} points for device {specific_device_id} over {unique_days} days...")

    # 6. Initialize the Map
    # Centers dynamically based on the specific device's average location
    center_lat = device_data['latitude'].mean()
    center_lon = device_data['longitude'].mean()
    
    # CartoDB positron provides a muted background so overlapping colored routes stand out
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB positron') 

    # 7. Define high-contrast colors for different days
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
              '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
              '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000']

    # 8. Group by Date and Plot
    grouped_by_day = device_data.groupby('date')
    
    # Create an interactive legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px; max-height: 400px; 
    overflow-y: auto; border:2px solid grey; z-index:9999; font-size:14px;
    background-color:white; padding: 10px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
    <b style="font-size:16px;">Vehicle Routing by Day</b><br>
    <hr style="margin: 5px 0;">
    '''
    
    color_index = 0
    for current_date, day_data in grouped_by_day:
        day_color = colors[color_index % len(colors)]
        day_str = str(current_date)
        
        # Add entry to legend
        legend_html += f'<div style="margin-bottom:4px;"><i style="background:{day_color}; width:14px; height:14px; float:left; margin-right:8px; border-radius:50%;"></i>{day_str} ({len(day_data)} pts)</div>'
        
        daily_coords = day_data[['latitude', 'longitude']].values.tolist()
        
        if len(daily_coords) > 1:
            # Draw the continuous route for the day
            folium.PolyLine(
                locations=daily_coords,
                color=day_color,
                weight=4,
                opacity=0.8,
                tooltip=f"Route: {day_str}"
            ).add_to(m)
            
            # Plot individual pings to identify stops and slow-moving traffic
            for _, row in day_data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=4,
                    color=day_color,
                    fill=True,
                    fill_opacity=1.0,
                    tooltip=f"<b>Date:</b> {day_str}<br><b>Time:</b> {row['time_only']}<br><b>Acc:</b> {row.get('horizontal_accuracy', 'N/A')}m"
                ).add_to(m)
                
        color_index += 1

    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    # 9. Save the Final Map
    output_filename = f"combined_multiday_{specific_device_id}.html"
    m.save(output_filename)
    print(f"\nSuccess! Open '{output_filename}' in your web browser to view the segmented trips.")

# ==========================================
# HOW TO USE
# ==========================================
# 1. Point this to the folder where your daily highly-active CSVs are stored
DAILY_FILES_FOLDER = './mumbai_active_data/*' 

# 2. Paste a specific device ID to track, or leave as None to map the most active vehicle
TARGET_DEVICE = None 

# Run the script
map_combined_daily_files(DAILY_FILES_FOLDER, TARGET_DEVICE)