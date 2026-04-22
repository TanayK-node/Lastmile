import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
import glob
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# This MUST match the folder from your pipe.py (Step 1 output)
REAL_DATA_FOLDER = "E:/clean_data/01_clean" 

MIN_LAT, MAX_LAT = 18.8900, 19.3000
MIN_LON, MAX_LON = 72.7500, 73.0000

def load_real_data(input_folder, max_rows_to_read=500000):
    """Loads ACTUAL data from your formatted CSVs."""
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"Error: No real data found in {input_folder}.")
        print("Make sure you have run Step 1 of pipe.py first!")
        return None

    print(f"Found {len(csv_files)} real data files. Loading a sample for visualization...")
    
    # We load a sample from the first few files to prevent memory crashes
    df_list = []
    rows_loaded = 0
    
    for file in csv_files:
        if rows_loaded >= max_rows_to_read:
            break
        try:
            # Read a chunk of real data
            chunk = pd.read_csv(file, nrows=100000) 
            df_list.append(chunk)
            rows_loaded += len(chunk)
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            
    df = pd.concat(df_list, ignore_index=True)
    
    # Standardize column names based on your raw data
    lat_col = 'lat' if 'lat' in df.columns else 'latitude'
    lon_col = 'lon' if 'lon' in df.columns else 'longitude'
    
    # Drop rows with missing coordinates
    df = df.dropna(subset=[lat_col, lon_col])
    
    print(f"Successfully loaded {len(df):,} REAL GPS pings.")
    return df, lat_col, lon_col

# ==========================================
# 2. STATIC SCATTER PLOT (BEFORE VS AFTER)
# ==========================================
def plot_real_before_after(df, lat_col, lon_col):
    print("Generating Real Data Scatter Plot...")
    
    # Apply the mathematical filter to your real data
    inside_box = (
        (df[lat_col] >= MIN_LAT) & (df[lat_col] <= MAX_LAT) &
        (df[lon_col] >= MIN_LON) & (df[lon_col] <= MAX_LON)
    )
    
    df_filtered = df[inside_box]
    df_dropped = df[~inside_box]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fig.suptitle('Real GPS Telemetry Filtering: Tanay - 2026 Pipeline', fontsize=16, fontweight='bold')

    # Subplot 1: Before Filtering (All India)
    axes[0].scatter(df[lon_col], df[lat_col], c='gray', alpha=0.3, s=2, label='Raw India Pings')
    rect0 = plt.Rectangle((MIN_LON, MIN_LAT), MAX_LON - MIN_LON, MAX_LAT - MIN_LAT, 
                          fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Mumbai Geofence')
    axes[0].add_patch(rect0)
    axes[0].set_title(f'Before Filtering (Real Data Sample: {len(df):,})', fontsize=14)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: After Filtering (Mumbai Only)
    axes[1].scatter(df_dropped[lon_col], df_dropped[lat_col], c='red', alpha=0.1, s=2, label='Dropped Pings (Outside)')
    axes[1].scatter(df_filtered[lon_col], df_filtered[lat_col], c='blue', alpha=0.4, s=2, label='Retained Pings (Mumbai)')
    rect1 = plt.Rectangle((MIN_LON, MIN_LAT), MAX_LON - MIN_LON, MAX_LAT - MIN_LAT, 
                          fill=False, edgecolor='black', linewidth=2)
    axes[1].add_patch(rect1)
    axes[1].set_title(f'After Filtering (Retained Mumbai Pings: {len(df_filtered):,})', fontsize=14)
    axes[1].set_xlabel('Longitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('real_mumbai_filtering_plot.png', dpi=300)
    print("  [+] Saved scatter plot as 'real_mumbai_filtering_plot.png'")
    plt.close()

# ==========================================
# 3. INTERACTIVE FOLIUM MAP (REAL DATA)
# ==========================================
def generate_real_interactive_map(df, lat_col, lon_col, max_points=3000):
    print("Generating Interactive Bounding Box Map with Real Data...")
    
    # Filter the real data
    df_filtered = df[
        (df[lat_col] >= MIN_LAT) & (df[lat_col] <= MAX_LAT) &
        (df[lon_col] >= MIN_LON) & (df[lon_col] <= MAX_LON)
    ]
    
    center_lat = (MIN_LAT + MAX_LAT) / 2
    center_lon = (MIN_LON + MAX_LON) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')

    # Draw the Bounding Box
    bounds = [[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]]
    folium.Rectangle(
        bounds=bounds,
        color='#ff0000',
        weight=3,
        fill=True,
        fill_color='#ff0000',
        fill_opacity=0.05,
        popup='Mumbai Operational Catchment'
    ).add_to(m)

    # Sample points so Folium doesn't crash your browser
    if len(df_filtered) > max_points:
        print(f"  * Downsampling from {len(df_filtered):,} to {max_points:,} points to prevent HTML crash.")
        df_sample = df_filtered.sample(n=max_points, random_state=42)
    else:
        df_sample = df_filtered

    # Plot the ACTUAL retained points
    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            weight=0
        ).add_to(m)

    output_html = "real_mumbai_bounding_box_map.html"
    m.save(output_html)
    print(f"  [+] Saved interactive map as '{output_html}'")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    result = load_real_data(REAL_DATA_FOLDER)
    if result:
        df, lat_col, lon_col = result
        plot_real_before_after(df, lat_col, lon_col)
        generate_real_interactive_map(df, lat_col, lon_col)