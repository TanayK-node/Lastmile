import random
import copy

def calculate_route_time(chromosome, stops, time_matrix, depot_idx=0, dwell_time=1.5):
    """
    Calculates the total time taken to complete the route sequence.
    Includes both the Network Drive Time and the Passenger Dwell Time.
    """
    total_time = 0
    current_node = depot_idx
    
    for stop_idx in chromosome:
        target_node = stops[stop_idx]['matrix_idx']
        
        # Drive time to the next stop
        total_time += time_matrix[current_node][target_node]
        
        # Idle time for boarding/alighting
        total_time += dwell_time 
        
        current_node = target_node
        
    # Return to the depot at the end of the loop
    total_time += time_matrix[current_node][depot_idx]
    
    return total_time

def order_crossover(parent1, parent2):
    """Standard OX1 Crossover to maintain valid routing sequences without duplicates."""
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = parent1[a:b+1]
    
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child: 
                p2_idx += 1
            child[i] = parent2[p2_idx]
    return child

def mutate(chromosome, current_mutation_rate):
    """Adaptive swap mutation."""
    if random.random() < current_mutation_rate:
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

def optimize_cluster_route(cluster_stops, time_matrix, dist_matrix, depot_idx=0, pop_size=50, generations=100):
    """
    STAGE 2: THE SEQUENCER (Traveling Salesperson Problem GA)
    Finds the optimal pathing for a pre-clustered geographic zone.
    """
    num_stops = len(cluster_stops)
    
    # If the cluster only has 1 or 2 stops, don't waste CPU on AI. Just return it.
    if num_stops <= 2:
        if num_stops == 0:
            return [], 0.0, 0.0

        if num_stops == 1:
            best_chromosome = [0]
        else:
            # For two stops, evaluate both orders and choose the faster loop.
            c1, c2 = [0, 1], [1, 0]
            t1 = calculate_route_time(c1, cluster_stops, time_matrix, depot_idx)
            t2 = calculate_route_time(c2, cluster_stops, time_matrix, depot_idx)
            best_chromosome = c1 if t1 <= t2 else c2

        ordered_route = [cluster_stops[i] for i in best_chromosome]
        best_time = calculate_route_time(best_chromosome, cluster_stops, time_matrix, depot_idx)

        final_distance = 0
        current_node = depot_idx
        for stop in ordered_route:
            final_distance += dist_matrix[current_node][stop['matrix_idx']]
            current_node = stop['matrix_idx']
        final_distance += dist_matrix[current_node][depot_idx]

        return ordered_route, best_time, final_distance
        
    # Initialize Population (Random sequences of the stops in this cluster)
    population = [random.sample(range(num_stops), num_stops) for _ in range(pop_size)]
    
    best_time = float('inf')
    best_chromosome = []
    
    # AGA Parameters
    base_mutation_rate = 0.10
    spike_mutation_rate = 0.50
    stagnation_limit = 10
    stagnation_counter = 0
    current_mutation_rate = base_mutation_rate
    
    for gen in range(generations):
        scored_population = []
        generation_improved = False
        
        for chromo in population:
            route_time = calculate_route_time(chromo, cluster_stops, time_matrix, depot_idx)
            scored_population.append((route_time, chromo))
            
            if route_time < best_time:
                best_time = route_time
                best_chromosome = chromo
                generation_improved = True
                
        # Adaptive Stagnation Logic
        if generation_improved:
            stagnation_counter = 0
            current_mutation_rate = base_mutation_rate
        else:
            stagnation_counter += 1
            if stagnation_counter >= stagnation_limit:
                current_mutation_rate = spike_mutation_rate
                
        # Sort and Select Elite
        scored_population.sort(key=lambda x: x[0])
        survivors = [item[1] for item in scored_population[:int(pop_size * 0.2)]]
        
        # Breed Next Generation
        next_gen = copy.deepcopy(survivors)
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = mutate(order_crossover(p1, p2), current_mutation_rate)
            next_gen.append(child)
            
        population = next_gen

    # Convert the winning chromosome into a list of actual stop dictionaries
    ordered_route = [cluster_stops[i] for i in best_chromosome]
    
    # Calculate the final operational distance (for the VKT metric)
    final_distance = 0
    current_node = depot_idx
    for stop in ordered_route:
        final_distance += dist_matrix[current_node][stop['matrix_idx']]
        current_node = stop['matrix_idx']
    final_distance += dist_matrix[current_node][depot_idx]
    
    return ordered_route, best_time, final_distance