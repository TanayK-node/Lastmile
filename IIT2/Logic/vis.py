import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
import os
import glob

def _format_duration_minutes(minutes):
    """Return duration as compact text, e.g. 2h 15m."""
    total_mins = int(round(float(minutes)))
    hours, mins = divmod(total_mins, 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def _build_visit_sessions_html(cluster_data, merge_gap_mins=20):
    """Build day-wise merged sessions for popup rendering."""
    cluster_data = cluster_data.sort_values('arrival_time').copy()
    visit_history_html = (
        "<div style='max-height: 180px; overflow-y: auto; border-top: 1px solid #ccc; "
        "padding-top: 6px; margin-top: 6px; font-size: 11px;'>"
    )

    for date, day_group in cluster_data.groupby(cluster_data['arrival_time'].dt.date):
        day_group = day_group.sort_values('arrival_time')
        sessions = []

        for _, stop_row in day_group.iterrows():
            start = stop_row['arrival_time']
            end = stop_row['departure_time']

            if not sessions:
                sessions.append({'start': start, 'end': end, 'stops': 1})
                continue

            last_session = sessions[-1]
            gap_mins = (start - last_session['end']).total_seconds() / 60.0

            if gap_mins <= merge_gap_mins:
                if end > last_session['end']:
                    last_session['end'] = end
                last_session['stops'] += 1
            else:
                sessions.append({'start': start, 'end': end, 'stops': 1})

        day_label = pd.to_datetime(date).strftime('%b %d, %Y')
        visit_history_html += f"<div style='margin-bottom: 6px;'><b>{day_label}</b></div>"

        day_total_mins = 0.0
        for session in sessions:
            session_mins = (session['end'] - session['start']).total_seconds() / 60.0
            day_total_mins += session_mins
            start_str = session['start'].strftime('%I:%M %p').lstrip('0')
            end_str = session['end'].strftime('%I:%M %p').lstrip('0')
            dur_str = _format_duration_minutes(session_mins)
            visit_history_html += (
                "<div style='margin-left: 8px; margin-bottom: 3px;'>"
                f"- {start_str} - {end_str} "
                f"<span style='color:#666;'>({dur_str}, {session['stops']} stop{'s' if session['stops'] != 1 else ''})</span>"
                "</div>"
            )

        visit_history_html += (
            f"<div style='margin-left: 8px; color:#444; margin-bottom: 6px;'>"
            f"Total: {_format_duration_minutes(day_total_mins)}"
            "</div>"
        )

    visit_history_html += "</div>"
    return visit_history_html

def identify_and_map_anchor_places_with_transit(stops_file, raw_data_folder, specific_device_id=None, cluster_radius_m=200):
    """
    Clusters anchor places (Home/Work) AND plots the raw moving GPS points 
    to visualize the actual transit routes taken.
    """
    print(f"Loading stop data from {stops_file}...")
    try:
        df = pd.read_csv(stops_file)
    except FileNotFoundError:
        print(f"Error: Could not find '{stops_file}'.")
        return

    df['arrival_time'] = pd.to_datetime(df['arrival_time'])
    df['departure_time'] = pd.to_datetime(df['departure_time'])
    
    # 1. Select the Device
    if specific_device_id is None:
        specific_device_id = df['device_aid'].value_counts().index[0]
        print(f"Auto-selected device: {specific_device_id}")
        
    user_stops = df[df['device_aid'] == specific_device_id].copy()
    user_stops = user_stops.sort_values('arrival_time').reset_index(drop=True)
    
    if len(user_stops) == 0:
        print("No data for this device.")
        return

    # ---------------------------------------------------------
    # NEW: LOAD RAW TRAJECTORY DATA FOR THIS SPECIFIC DEVICE
    # ---------------------------------------------------------
    print(f"Loading raw transit data for device {specific_device_id}...")
    search_pattern = os.path.join(raw_data_folder, "**", "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    raw_data_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            # Immediately filter for just this device to save memory
            temp_df = temp_df[temp_df['device_aid'] == specific_device_id]
            if not temp_df.empty:
                raw_data_list.append(temp_df)
        except Exception:
            pass

    has_raw_data = False
    if raw_data_list:
        raw_df = pd.concat(raw_data_list, ignore_index=True)
        raw_df = raw_df.sort_values('timestamp')
        has_raw_data = True
        print(f"Found {len(raw_df):,} raw GPS points for transit mapping.")
    else:
        print("Warning: No raw trajectory data found for this device.")

    # ---------------------------------------------------------

    print(f"\nApplying DBSCAN clustering to {len(user_stops)} individual stops...")

    # 2. Apply DBSCAN Spatial Clustering
    coords = np.radians(user_stops[['stop_lat', 'stop_lon']])
    kms_per_radian = 6371.0088
    epsilon = (cluster_radius_m / 1000.0) / kms_per_radian 
    
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(coords)
    user_stops['place_cluster_id'] = db.labels_

    # 3. Aggregate the Clusters into "Master Places"
    master_places = []
    grouped_clusters = user_stops.groupby('place_cluster_id')
    
    for cluster_id, cluster_data in grouped_clusters:
        centroid_lat = cluster_data['stop_lat'].mean()
        centroid_lon = cluster_data['stop_lon'].mean()
        
        total_duration = cluster_data['duration_mins'].sum()
        total_pings = cluster_data['ping_count'].sum()
        visit_count = len(cluster_data)
        primary_purpose = cluster_data['likely_purpose'].mode()[0]
        
        visit_history_html = _build_visit_sessions_html(cluster_data, merge_gap_mins=20)

        master_places.append({
            'place_id': cluster_id,
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'total_duration_mins': total_duration,
            'total_visits': visit_count,
            'total_pings': total_pings,
            'primary_purpose': primary_purpose,
            'visit_history_html': visit_history_html  
        })
        
    places_df = pd.DataFrame(master_places)
    print(f"Successfully condensed into {len(places_df)} Master Places!")

    # 4. Map the Master Places and the Network
    center_lat = places_df['centroid_lat'].mean()
    center_lon = places_df['centroid_lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

    # --- NEW: PLOT ACTUAL TRANSIT PATH ---
    if has_raw_data:
        raw_coords = raw_df[['latitude', 'longitude']].values.tolist()
        
        # 1. Draw the continuous route line based on raw pings
        folium.PolyLine(
            locations=raw_coords,
            color='#9370DB', # Purple color for transit path
            weight=3,
            opacity=0.7,
            tooltip="Actual Transit Trace"
        ).add_to(m)
        
        # 2. Add tiny dots for every single GPS ping
        for lat, lon in raw_coords:
            folium.CircleMarker(
                location=[lat, lon],
                radius=1.5,
                color='black',
                fill=True,
                fill_color='black',
                fill_opacity=0.6
            ).add_to(m)
    else:
        # Fallback to straight lines if raw data isn't found
        travel_sequence_coords = []
        for cluster_id in user_stops['place_cluster_id']:
            place_info = places_df[places_df['place_id'] == cluster_id].iloc[0]
            travel_sequence_coords.append([place_info['centroid_lat'], place_info['centroid_lon']])

        folium.PolyLine(
            locations=travel_sequence_coords,
            color='gray', weight=2, opacity=0.6, tooltip="Abstract Network"
        ).add_to(m)

    # Plot the clean Master Places (Anchors)
    for _, row in places_df.iterrows():
        purpose = row['primary_purpose']
        
        if 'Home' in purpose:
            marker_color = 'blue'
            icon_type = 'home'
        elif 'Work' in purpose:
            marker_color = 'green'
            icon_type = 'briefcase'
        else:
            marker_color = 'orange'
            icon_type = 'info-sign'

        popup_html = f"""
        <div style="width:260px; font-family:Arial, sans-serif;">
            <h4 style="margin-bottom:5px; margin-top:0px; color:{marker_color};">{purpose}</h4>
            <b>Total Visits:</b> {row['total_visits']} times<br>
            <b>Total Time Spent:</b> {round(row['total_duration_mins'] / 60, 1)} hours<br>
            <b>Visit Log:</b>
            {row['visit_history_html']}
        </div>
        """

        folium.Marker(
            location=[row['centroid_lat'], row['centroid_lon']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{purpose} (Visited {row['total_visits']}x)",
            icon=folium.Icon(color=marker_color, icon=icon_type)
        ).add_to(m)

    # Save Output
    output_filename = f"detailed_network_with_transit_{specific_device_id}.html"
    m.save(output_filename)
    print(f"\nSuccess! Map with actual transit traces saved as: {output_filename}")

# ==========================================
# HOW TO USE
# ==========================================
STOPS_FILE = 'mumbai_multiday_stops_robust.csv' 
RAW_DATA_FOLDER = '../mumbai_active_data/*'  # <-- NEW: Point to your raw data folder
TARGET_DEVICE = '00007cff-4a4a-61e7-b4a0-77f3d06cd01b'

# Run it!
identify_and_map_anchor_places_with_transit(STOPS_FILE, RAW_DATA_FOLDER, TARGET_DEVICE, cluster_radius_m=200)