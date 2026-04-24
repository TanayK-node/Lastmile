import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import os

def run_dbscan_sensitivity():
    print("==========================================")
    print("INITIATING DBSCAN DENSITY SENSITIVITY SWEEP")
    print("==========================================\n")
    
    # Load all demand to get a city-wide average
    try:
        dfs = [
            pd.read_csv("./demand_matrices/AM_feeder_demand.csv"),
            pd.read_csv("./demand_matrices/AM_dispersal_demand.csv"),
            pd.read_csv("./demand_matrices/PM_feeder_demand.csv"),
            pd.read_csv("./demand_matrices/PM_dispersal_demand.csv")
        ]
        city_demand = pd.concat(dfs, ignore_index=True)
    except FileNotFoundError:
        print("Error: Demand CSVs not found in ./demand_matrices/")
        return

    total_commuters = len(city_demand)
    print(f"Total City-Wide Commuters Loaded: {total_commuters}")
    
    # We will test minimum cluster sizes from 2 people up to 8 people
    min_samples_to_test = [2, 3, 4, 5, 6, 7, 8]
    results = []
    
    coords = np.radians(city_demand[['lat', 'lon']])
    epsilon = (300 / 1000.0) / 6371.0088  # 300 meters fixed radius

    for min_pax in min_samples_to_test:
        # Run DBSCAN
        db = DBSCAN(eps=epsilon, min_samples=min_pax, algorithm='ball_tree', metric='haversine').fit(coords)
        city_demand['cluster'] = db.labels_
        
        # Calculate Metrics
        noise_points = len(city_demand[city_demand['cluster'] == -1])
        covered_points = total_commuters - noise_points
        coverage_percent = (covered_points / total_commuters) * 100
        
        valid_clusters = city_demand[city_demand['cluster'] != -1]
        num_virtual_stops = valid_clusters['cluster'].nunique()
        
        # Estimate the Operator Cost impact:
        # More stops = more complex routes = exponentially higher VKT
        
        results.append({
            'Min Pax per Stop': min_pax,
            'System Coverage (%)': coverage_percent,
            'Unserviced Noise (%)': (noise_points / total_commuters) * 100,
            'Virtual Stops Generated': num_virtual_stops
        })
        
        print(f"Threshold: {min_pax} Pax | Coverage: {coverage_percent:.1f}% | Stops Required: {num_virtual_stops}")

    results_df = pd.DataFrame(results)

    # ==========================================
    # ACADEMIC PLOTTING: THE DUAL-AXIS PARETO
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot 1: Commuter Coverage (Bar Chart)
    color = '#3498db'
    ax1.set_xlabel('Density Constraint: Minimum Passengers Required per Stop', fontweight='bold')
    ax1.set_ylabel('System Coverage (%)', color=color, fontweight='bold')
    bars = ax1.bar(results_df['Min Pax per Stop'], results_df['System Coverage (%)'], color=color, alpha=0.7, label='Commuter Coverage %')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 105)

    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

    # Plot 2: Operator Cost / Stops Generated (Line Chart on secondary Y axis)
    ax2 = ax1.twinx()
    color = '#e74c3c'
    ax2.set_ylabel('Operator Cost: Virtual Stops Required', color=color, fontweight='bold')
    ax2.plot(results_df['Min Pax per Stop'], results_df['Virtual Stops Generated'], color=color, marker='o', linewidth=3, markersize=8, label='Virtual Stops Generated')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Highlight the chosen parameter (min_samples = 4)
    ax1.axvline(x=4, color='#2ecc71', linestyle='--', linewidth=2, label='Chosen Inflection Point (k=4)')

    plt.title('DBSCAN Parameter Optimization: System Coverage vs. Operator Cost', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

    plt.tight_layout()
    output_img = "dbscan_sensitivity_ctsem.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nAnalysis Complete! Saved Pareto graph to {output_img}")

if __name__ == "__main__":
    run_dbscan_sensitivity()