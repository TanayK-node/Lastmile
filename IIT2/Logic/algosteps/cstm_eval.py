import pandas as pd
import numpy as np
import os
import json

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def generate_lr_metrics(feeder_csv, dispersal_csv, stations_dict, routing_summary_file, bus_capacity=30):
    print("==========================================")
    print("AUTOMATED CTSEM ACADEMIC EVALUATION")
    print("==========================================\n")
    
    # 1. Automatically load the AI routing metrics
    try:
        with open(routing_summary_file, "r") as f:
            ai_data = json.load(f)
            total_buses = ai_data["total_buses"]
            total_mileage_km = ai_data["total_mileage_km"]
    except FileNotFoundError:
        print(f"[!] Error: Could not find '{routing_summary_file}'. Run master_routing_engine.py first.")
        return

    # 2. Extract Spatial Demand Metrics
    total_commuters = 0
    all_distances = []

    if os.path.exists(feeder_csv):
        df_feeder = pd.read_csv(feeder_csv)
        total_commuters += len(df_feeder)
        for _, row in df_feeder.iterrows():
            s_lat, s_lon = stations_dict[row['station']]
            all_distances.append(calculate_distance(row['lat'], row['lon'], s_lat, s_lon))

    if os.path.exists(dispersal_csv):
        df_dispersal = pd.read_csv(dispersal_csv)
        total_commuters += len(df_dispersal)
        for _, row in df_dispersal.iterrows():
            s_lat, s_lon = stations_dict[row['station']]
            all_distances.append(calculate_distance(row['lat'], row['lon'], s_lat, s_lon))

    # 3. Mathematical Calculations
    avg_transfer_distance = np.mean(all_distances) / 1000 if all_distances else 0
    max_accessibility_radius = np.max(all_distances) / 1000 if all_distances else 0
    
    # Capacity Utilization Calculation (Fleet Efficiency)
    total_available_seats = total_buses * bus_capacity
    capacity_utilization = (total_commuters / total_available_seats) * 100 if total_available_seats > 0 else 0

    # 4. Compile Literature-Backed Table
    metrics_table = {
        "Evaluation Metric": [
            "Total Validated Commuters Served",
            "Avg. Access/Egress Distance (Sun & Meng, 2026 proxy)",
            "Max Catchment Accessibility Radius (Jiang et al.)",
            "Total System Routing Mileage",
            "Optimized Fleet Size (CVRP Output)",
            "Fleet Capacity Utilization Rate"
        ],
        "Result": [
            f"{total_commuters:,}",
            f"{avg_transfer_distance:.2f} km",
            f"{max_accessibility_radius:.2f} km",
            f"{total_mileage_km:.2f} km",
            f"{total_buses} Buses",
            f"{capacity_utilization:.1f}%"
        ]
    }

    results_df = pd.DataFrame(metrics_table)
    print(results_df.to_markdown(index=False))
    
    results_df.to_csv("ctsem_final_paper_metrics.csv", index=False)
    print("\n[+] Table successfully saved to 'ctsem_final_paper_metrics.csv'")

if __name__ == "__main__":
    # File Paths
    FEEDER_FILE = "./demand_matrices/first_mile_feeder_demand.csv"
    DISPERSAL_FILE = "./demand_matrices/last_mile_dispersal_demand.csv"
    ROUTING_SUMMARY = "routing_summary.json"
    
    # Target Stations
    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }
    
    generate_lr_metrics(FEEDER_FILE, DISPERSAL_FILE, STATIONS, ROUTING_SUMMARY)