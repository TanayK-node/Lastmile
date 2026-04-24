import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import os

def run_sensitivity_analysis(demand_df):
    print("Starting DBSCAN Sensitivity Analysis...")
    
    # 1. Define the Parameter Grid to Test
    eps_meters_list = [100, 200, 300, 400, 500]  # Walking distances in meters
    min_samples_list = [2, 4, 6, 8, 10]          # Minimum pax to justify a hub
    
    # Prepare coordinates
    coords = np.radians(demand_df[['lat', 'lon']])
    
    results = []
    
    total_commuters = len(demand_df)
    
    # 2. Iterate through every combination
    for eps_m in eps_meters_list:
        for min_samp in min_samples_list:
            # Convert meters to radians for Haversine
            epsilon_rad = (eps_m / 1000.0) / 6371.0088
            
            db = DBSCAN(eps=epsilon_rad, min_samples=min_samp, 
                        algorithm='ball_tree', metric='haversine').fit(coords)
            
            labels = db.labels_
            
            # 3. Calculate Metrics
            # Hubs are any label >= 0. Noise is label -1.
            num_hubs = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = list(labels).count(-1)
            noise_ratio = (noise_points / total_commuters) * 100 if total_commuters > 0 else 0
            
            # Average passengers per valid hub
            valid_labels = [l for l in labels if l != -1]
            avg_pax = len(valid_labels) / num_hubs if num_hubs > 0 else 0
            
            results.append({
                'Epsilon (m)': eps_m,
                'Min Samples': min_samp,
                'Number of Hubs': num_hubs,
                'Noise Ratio (%)': noise_ratio,
                'Avg Pax per Hub': avg_pax
            })
            
    results_df = pd.DataFrame(results)
    
    # 4. Generate Academic Heatmaps
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap 1: Number of Virtual Hubs Generated
    pivot_hubs = results_df.pivot(index='Epsilon (m)', columns='Min Samples', values='Number of Hubs')
    sns.heatmap(pivot_hubs, annot=True, fmt=".0f", cmap="YlGnBu", ax=axes[0], cbar_kws={'label': 'Virtual Hubs'})
    axes[0].set_title('Impact of Parameters on Virtual Hub Generation', fontsize=14)
    axes[0].invert_yaxis() # Put lowest distances at the bottom
    
    # Heatmap 2: Percentage of Unserviced Commuters (Noise)
    pivot_noise = results_df.pivot(index='Epsilon (m)', columns='Min Samples', values='Noise Ratio (%)')
    sns.heatmap(pivot_noise, annot=True, fmt=".1f", cmap="OrRd", ax=axes[1], cbar_kws={'label': '% Unserviced (Noise)'})
    axes[1].set_title('Impact of Parameters on Unserviced Demand (Noise %)', fontsize=14)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    output_filename = "dbscan_sensitivity_ctsem.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Analysis Complete! Graph saved as {output_filename}")
    
    return results_df

if __name__ == "__main__":
    # Load your actual processed telemetry data
    try:
        # Test on one major station's demand to get a localized sensitivity curve
        df = pd.read_csv("./demand_matrices/first_mile_feeder_demand.csv")
        station_df = df[df['station'] == "Andheri"].copy()
        
        if not station_df.empty:
            run_sensitivity_analysis(station_df)
        else:
            print("Error: No demand found for Andheri in the CSV.")
            
    except FileNotFoundError:
        print("Error: Could not find demand_matrices/first_mile_feeder_demand.csv.")