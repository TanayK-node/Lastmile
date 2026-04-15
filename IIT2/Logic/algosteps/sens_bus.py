import pandas as pd
import numpy as np
import random
import copy
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CORE GA FUNCTIONS
# ==========================================
def get_distance(lat1, lon1, lat2, lon2):
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def build_matrix(s_lat, s_lon, stops):
    pts = [{'lat': s_lat, 'lon': s_lon}] + stops
    n = len(pts)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: matrix[i][j] = get_distance(pts[i]['lat'], pts[i]['lon'], pts[j]['lat'], pts[j]['lon'])
    return matrix

def preprocess(stops_df, cap):
    processed = []
    idx = 1 
    for _, row in stops_df.iterrows():
        dem = row['unique_commuters']
        while dem > 0:
            load = min(dem, cap)
            processed.append({
                'matrix_idx': idx, 
                'lat': row['lat'], 
                'lon': row['lon'], 
                'load': load
            })
            dem -= load
            idx += 1
    return processed

def decode(chromo, stops, cap, dist_matrix):
    """Calculates both the Distance AND the Number of Buses needed."""
    load, dist, last = 0, 0, 0
    buses_needed = 1
    
    for s_idx in chromo:
        stop = stops[s_idx]
        if load + stop['load'] > cap:
            dist += dist_matrix[last][0] 
            load, last = 0, 0
            buses_needed += 1  # Dispatch a new bus
            
        load += stop['load']
        dist += dist_matrix[last][stop['matrix_idx']]
        last = stop['matrix_idx']
        
    dist += dist_matrix[last][0] # Final return to station
    return dist, buses_needed

def ox1(p1, p2):
    sz = len(p1)
    a, b = sorted(random.sample(range(sz), 2))
    child = [-1] * sz
    child[a:b+1] = p1[a:b+1]
    p2_idx = 0
    for i in range(sz):
        if child[i] == -1:
            while p2[p2_idx] in child: p2_idx += 1
            child[i] = p2[p2_idx]
    return child

def mutate(chromo, rate=0.2):
    if random.random() < rate:
        a, b = random.sample(range(len(chromo)), 2)
        chromo[a], chromo[b] = chromo[b], chromo[a]
    return chromo

# ==========================================
# 2. THE 3D GRID SEARCH ENGINE
# ==========================================
def run_comprehensive_search(stops_file, s_lat, s_lon):
    print("Loading Virtual Stops...")
    df = pd.read_csv(stops_file)
    
    # Define the 3D Hyperparameter Space
    capacities = [15, 30, 50]       # Mini-bus vs Standard vs Large
    populations = [50, 100, 200]    # AI Gene Pool Size
    generations_list = [50, 150, 300] # AI Evolution Time
    
    results = []
    total_runs = len(capacities) * len(populations) * len(generations_list)
    current_run = 1

    print(f"\nStarting 3D Grid Search: {total_runs} Combinations...")
    print("="*75)
    
    for cap in capacities:
        # Preprocess demand based on this specific bus capacity
        processed = preprocess(df, cap)
        dist_matrix = build_matrix(s_lat, s_lon, processed)
        num_stops = len(processed)
        
        for pop in populations:
            for gen in generations_list:
                print(f"Run {current_run}/{total_runs} | Cap: {cap}, Pop: {pop}, Gen: {gen}...", end=" ")
                start_time = time.time()
                
                population = [random.sample(range(num_stops), num_stops) for _ in range(pop)]
                best_dist = float('inf')
                best_buses = 0
                
                for _ in range(gen):
                    scored = []
                    for chromo in population:
                        d, b = decode(chromo, processed, cap, dist_matrix)
                        scored.append((d, chromo, b))
                        if d < best_dist: 
                            best_dist = d
                            best_buses = b
                            
                    scored.sort(key=lambda x: x[0])
                    survivors = [item[1] for item in scored[:int(pop * 0.2)]]
                    
                    next_gen = copy.deepcopy(survivors)
                    while len(next_gen) < pop:
                        p1, p2 = random.sample(survivors, 2)
                        next_gen.append(mutate(ox1(p1, p2)))
                    population = next_gen
                    
                elapsed = round(time.time() - start_time, 2)
                dist_km = round(best_dist / 1000, 2)
                print(f"Dist: {dist_km}km | Buses: {best_buses} | Time: {elapsed}s")
                
                results.append({
                    "Capacity": cap,
                    "Population": pop,
                    "Generations": gen,
                    "Distance (km)": dist_km,
                    "Buses Needed": best_buses,
                    "Compute Time (s)": elapsed
                })
                current_run += 1

    # ==========================================
    # 3. DATA ANALYSIS & VISUALIZATION
    # ==========================================
    results_df = pd.DataFrame(results)
    
    # Plotting Dashboard
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Plot 1: The Pareto Front (Business Logic) ---
    sns.scatterplot(data=results_df, x="Buses Needed", y="Distance (km)", hue="Capacity", 
                    palette="Set1", s=150, ax=axes[0], alpha=0.8)
    axes[0].set_title("Pareto Front: Fleet Size vs. Mileage\n(Operational Cost Trade-off)", fontweight='bold')
    axes[0].set_xlabel("Total Buses Dispatched (Labor Cost)")
    axes[0].set_ylabel("Total Route Mileage in km (Fuel Cost)")
    
    # --- Plot 2 & 3: Heatmaps for AI Tuning ---
    # We average the Distance and Time across all capacities to see pure AI performance
    dist_pivot = results_df.pivot_table(index="Population", columns="Generations", values="Distance (km)", aggfunc='mean')
    time_pivot = results_df.pivot_table(index="Population", columns="Generations", values="Compute Time (s)", aggfunc='mean')
    
    sns.heatmap(dist_pivot, annot=True, fmt=".1f", cmap="YlGnBu_r", ax=axes[1], cbar_kws={'label': 'Avg Fleet Mileage (km)'})
    axes[1].set_title("AI Route Optimization Score\n(Population vs Generations)", fontweight='bold')
    
    sns.heatmap(time_pivot, annot=True, fmt=".1f", cmap="OrRd", ax=axes[2], cbar_kws={'label': 'Avg Compute Time (s)'})
    axes[2].set_title("Computational Cost\n(Hardware Constraint)", fontweight='bold')
    
    plt.suptitle("Master Urban Transit Grid Search: Operations & AI Hyperparameters", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("master_grid_search_dashboard.png", dpi=300, bbox_inches='tight')
    print("\n" + "="*75)
    print("Success! The Master Dashboard has been saved to 'master_grid_search_dashboard.png'")
    print("="*75)

# Run it!
STOPS_FILE = 'smart_virtual_stops_checkpoint.csv' 
STATION_LAT = 19.1197
STATION_LON = 72.8464

run_comprehensive_search(STOPS_FILE, STATION_LAT, STATION_LON)