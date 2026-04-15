import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

def analyze_device_pings(input_folder, output_folder, min_points_threshold=20):
    """
    Combines all Mumbai CSVs, calculates how many points each device has,
    plots the distribution, and saves a cleaned dataset removing low-ping devices.
    """
    # 1. Load all Mumbai CSV files
    search_pattern = os.path.join(input_folder, "mumbai_only_*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No files found in {input_folder}")
        return
        
    print(f"Loading {len(csv_files)} Mumbai files...")
    
    # Combine them into one giant dataframe
    all_data = []
    for f in csv_files:
        df = pd.read_csv(f)
        all_data.append(df)
        
    master_df = pd.concat(all_data, ignore_index=True)
    total_original_rows = len(master_df)
    
    # 2. Count points per device
    print("\nCalculating points per device...")
    device_counts = master_df['device_aid'].value_counts()
    
    # Calculate basic statistics
    avg_points = device_counts.mean()
    median_points = device_counts.median()
    max_points = device_counts.max()
    
    print("\n" + "="*40)
    print("DEVICE ACTIVITY STATISTICS")
    print("="*40)
    print(f"Total Unique Devices: {len(device_counts):,}")
    print(f"Average points per device: {avg_points:.2f}")
    print(f"Median points per device:  {median_points:.0f} (50% of devices have this or fewer)")
    print(f"Max points for one device: {max_points:,}")
    print("="*40)

    # 3. Plot the Graph
    print("\nGenerating Distribution Graph...")
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 2 subplots (one standard, one zoomed-in/log scale)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: Standard Histogram (Capped at 500 points for readability)
    capped_data = device_counts[device_counts <= 500]
    sns.histplot(capped_data, bins=50, ax=axes[0], color='blue', kde=False)
    axes[0].set_title('Distribution of Pings per Device (0 - 500 pings)')
    axes[0].set_xlabel('Number of Pings')
    axes[0].set_ylabel('Number of Devices')
    
    # Plot B: Cumulative Distribution to help choose a threshold
    sns.ecdfplot(data=device_counts, ax=axes[1], color='red')
    axes[1].set_title('Cumulative Percentage of Devices')
    axes[1].set_xlabel('Number of Pings')
    axes[1].set_ylabel('Percentage of Devices')
    axes[1].set_xlim(0, 100) # Zoom in on the first 100 points
    
    # Add a line showing your chosen threshold
    axes[1].axvline(min_points_threshold, color='black', linestyle='--', label=f'Threshold: {min_points_threshold} pings')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('device_ping_distribution.png', dpi=300)
    plt.close()
    print("Graph saved as 'device_ping_distribution.png'.")

    # 4. Filter out the low-ping devices
    print(f"\nFiltering out devices with fewer than {min_points_threshold} pings...")
    
    # Get a list of devices that meet the threshold
    valid_devices = device_counts[device_counts >= min_points_threshold].index
    
    # Keep only rows belonging to those valid devices
    filtered_df = master_df[master_df['device_aid'].isin(valid_devices)]
    
    total_kept_rows = len(filtered_df)
    devices_kept = len(valid_devices)
    
    print(f"Devices kept: {devices_kept:,} out of {len(device_counts):,}")
    print(f"Data rows kept: {total_kept_rows:,} out of {total_original_rows:,}")
    
    # 5. Save the Highly Active Data
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_file = os.path.join(output_folder, "mumbai_highly_active_devices.csv")
    
    # Sort by device and then by time so it's perfectly ordered for mapping
    filtered_df = filtered_df.sort_values(['device_aid', 'timestamp'])
    filtered_df.to_csv(output_file, index=False)
    
    print(f"\nSuccess! Cleaned, highly-active dataset saved to: {output_file}")

# ==========================================
# HOW TO USE
# ==========================================
# 1. Folder containing your "mumbai_only_*.csv" files
INPUT_DIR = './mumbai_time/01_time/' 

# 2. Folder where the final filtered master file will go
OUTPUT_DIR = '../mumbai_active_data/01_time'

# 3. SET YOUR THRESHOLD (e.g., ignore devices with fewer than 20 points across the 24 hours)
MINIMUM_PINGS_REQUIRED = 20

# Run it!
analyze_device_pings(INPUT_DIR, OUTPUT_DIR, MINIMUM_PINGS_REQUIRED)