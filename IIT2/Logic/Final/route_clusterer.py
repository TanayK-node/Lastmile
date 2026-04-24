import pandas as pd
import numpy as np
import random
import copy

def calculate_dynamic_fitness(chromosome, virtual_stops_list, time_matrix, max_cluster_time=25.0):
    """
    MULTI-OBJECTIVE FITNESS FUNCTION
    Dynamically balances Fleet Size vs. Travel Time without hardcoding the number of stops.
    """
    # 1. Identify how many unique routes the AI actually decided to use
    active_zones = set(chromosome)
    fleet_size = len(active_zones)
    
    # COST 1: The Fleet Penalty (Operator Cost)
    # We penalize the AI for every bus it decides to dispatch. 
    # This forces it to compress stops into fewer routes organically.
    fleet_penalty = fleet_size * 500 
    
    compactness_cost = 0
    time_violation_penalty = 0
    
    zones = {z: [] for z in active_zones}
    for idx, zone_id in enumerate(chromosome):
        matrix_idx = virtual_stops_list[idx]['matrix_idx']
        zones[zone_id].append(matrix_idx)
        
    # COST 2: The Time Constraint (Commuter Cost)
    for zone_id, indices in zones.items():
        if len(indices) > 1:
            zone_pairwise_time = 0
            
            # Calculate the topological spread of this specific zone
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    time_cost = time_matrix[indices[i]][indices[j]]
                    zone_pairwise_time += time_cost
                    compactness_cost += time_cost
            
            # If the AI squeezed too many stops together and the road time 
            # exceeds the threshold, hit it with a massive "kill" penalty.
            if zone_pairwise_time > max_cluster_time:
                time_violation_penalty += (zone_pairwise_time - max_cluster_time) * 10000
                
    return fleet_penalty + compactness_cost + time_violation_penalty

def uniform_crossover(parent1, parent2):
    child = []
    for i in range(len(parent1)):
        child.append(parent1[i] if random.random() < 0.5 else parent2[i])
    return child

def mutate(chromosome, max_possible_zones, mutation_rate=0.15):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(0, max_possible_zones - 1)
    return chromosome

def cluster_hubs_into_routes(virtual_stops_df, time_matrix, pop_size=100, generations=100):
    """
    STAGE 1: DYNAMIC GENETIC CLUSTERER
    Lets the AI figure out the perfect number of routes dynamically.
    Note: We removed the 'max_stops_per_route' parameter entirely!
    """
    if virtual_stops_df is None or virtual_stops_df.empty:
        return None
        
    virtual_stops = virtual_stops_df.copy()
    virtual_stops_list = virtual_stops.to_dict('records')
    num_stops = len(virtual_stops)
    
    # If there are only 1 or 2 stops, it's obviously 1 route.
    if num_stops <= 2:
        virtual_stops['route_zone_id'] = 0
        print(f"  [*] Stage 1 GA: Demand is low. Grouped {num_stops} stops into 1 Route Zone.")
        return virtual_stops

    # We give the AI the maximum possible freedom: 
    # It can choose to use anywhere from 1 route up to (num_stops - 1) routes.
    max_possible_zones = max(2, int(num_stops * 0.8))
    print(f"  [*] Stage 1 GA: Dynamically optimizing Fleet Size for {num_stops} stops...")
    
    population = []
    for _ in range(pop_size):
        chromo = [random.randint(0, max_possible_zones - 1) for _ in range(num_stops)]
        population.append(chromo)
        
    best_fitness = float('inf')
    best_chromosome = []
    
    for gen in range(generations):
        scored_population = []
        for chromo in population:
            # We enforce a strict 25-minute topological time limit per zone
            fitness = calculate_dynamic_fitness(chromo, virtual_stops_list, time_matrix, max_cluster_time=25.0)
            scored_population.append((fitness, chromo))
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_chromosome = chromo
                
        scored_population.sort(key=lambda x: x[0])
        survivors = [item[1] for item in scored_population[:int(pop_size * 0.2)]]
        
        next_gen = copy.deepcopy(survivors)
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = mutate(uniform_crossover(p1, p2), max_possible_zones)
            next_gen.append(child)
            
        population = next_gen

    # Apply the winning genes
    virtual_stops['route_zone_id'] = best_chromosome
    
    # Print the AI's final decision
    active_zones = virtual_stops['route_zone_id'].unique()
    print(f"  [*] Stage 1 GA Decision: Compressed into {len(active_zones)} optimal Route Zones.")
    
    zone_counts = virtual_stops['route_zone_id'].value_counts()
    for zone, count in zone_counts.items():
        print(f"      -> Zone ID {zone}: {count} stops assigned")
        
    return virtual_stops