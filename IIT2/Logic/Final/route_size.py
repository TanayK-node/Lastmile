import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IIT2.Logic.Final.osmnx_router as oxr

def nearest_neighbor_time(stops_subset, time_matrix, depot_idx=0):
    """Calculates the absolute fastest path through a subset of stops."""
    unvisited = stops_subset.copy()
    current_node = depot_idx
    total_drive_time = 0
    
    while unvisited:
        closest_stop = None
        min_time = float('inf')
        
        for stop in unvisited:
            drive_time = time_matrix[current_node][stop['matrix_idx']]
            if drive_time < min_time:
                min_time = drive_time
                closest_stop = stop
                
        total_drive_time += min_time
        current_node = closest_stop['matrix_idx']
        unvisited.remove(closest_stop)
        
    # Return to depot
    total_drive_time += time_matrix[current_node][depot_idx]
    return total_drive_time

def run_stop_limit_analysis():
    print("Loading Major Roads Graph...")
    G = oxr.load_or_download_mumbai_graph()
    
    # We will simulate a large pool of 20 random stops around Andheri
    # Note: Replace this block with your actual Andheri virtual stops dataframe
    # For this script to run standalone, we are using dummy stop data
    # that assumes a matrix index 1 through 20.
    
    # Simulate a time matrix where average travel between adjacent stops is ~3 mins
    # (In your actual code, pass the real time_matrix from osmnx_router)
    pool_size = 20
    np.random.seed(42)
    mock_time_matrix = np.random.uniform(1.5, 4.5, (pool_size + 1, pool_size + 1))
    
    # -----------------------------------------
    # PARAMETERS FOR THE PAPER
    # -----------------------------------------
    DWELL_TIME_PER_STOP_MINS = 1.5 
    MAX_ACCEPTABLE_RIDE_TIME = 45.0 # Commuters won't tolerate a ride longer than 45 mins
    
    stop_counts_to_test = [3, 4, 5, 6, 8, 10, 12, 15]
    total_times = []
    drive_times = []
    dwell_times = []
    
    print("Simulating Route Times for varying Cluster Sizes...")
    
    for k in stop_counts_to_test:
        # Create a mock route of 'k' stops
        subset = [{'matrix_idx': i} for i in range(1, k + 1)]
        
        # Calculate optimal drive time
        drive_time = nearest_neighbor_time(subset, mock_time_matrix)
        
        # Calculate idle time
        dwell_time = k * DWELL_TIME_PER_STOP_MINS
        
        total_time = drive_time + dwell_time
        
        drive_times.append(drive_time)
        dwell_times.append(dwell_time)
        total_times.append(total_time)
        
    # -----------------------------------------
    # ACADEMIC PLOTTING
    # -----------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Stacked bar/line hybrid for clear breakdown
    plt.plot(stop_counts_to_test, total_times, marker='o', color='black', linewidth=2, label='Total Route Time')
    plt.bar(stop_counts_to_test, drive_times, color='#3498db', alpha=0.7, label='Network Drive Time')
    plt.bar(stop_counts_to_test, dwell_times, bottom=drive_times, color='#e74c3c', alpha=0.7, label='Passenger Dwell Time')
    
    # The crucial threshold line
    plt.axhline(y=MAX_ACCEPTABLE_RIDE_TIME, color='red', linestyle='--', linewidth=2, label='Max Acceptable Commute (45 min)')
    
    plt.title("Route Size Threshold Analysis: Drive Time vs. Dwell Time", fontsize=14)
    plt.xlabel("Number of Virtual Hubs per Route", fontsize=12)
    plt.ylabel("Total Loop Time (Minutes)", fontsize=12)
    plt.xticks(stop_counts_to_test)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_filename = "route_size_threshold_ctsem.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graph Saved: {output_filename}")

if __name__ == "__main__":
    run_stop_limit_analysis()