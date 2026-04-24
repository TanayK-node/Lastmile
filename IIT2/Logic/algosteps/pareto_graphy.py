import matplotlib.pyplot as plt
import pandas as pd
import route2 as re
import IIT2.Logic.Final.osmnx_router as oxr

print("Loading Mumbai Graph for Pareto Analysis...")
G = oxr.load_or_download_mumbai_graph()

# Grab just one station's morning demand to test
feeder_df = pd.read_csv("./demand_matrices/first_mile_feeder_demand.csv")
station_feeder = feeder_df[feeder_df['station'] == "Andheri"]
s_lat, s_lon = 19.1197, 72.8464
feeder_stops = re.generate_virtual_stops(station_feeder, s_lat, s_lon)

alphas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
operator_costs = []
commuter_costs = []

print("Running Multi-Objective AI across 6 different weight scenarios...")
for alpha in alphas:
    print(f"  -> Testing Alpha = {alpha}...")
    routes, dist, ride = re.run_genetic_ai(G, feeder_stops, s_lat, s_lon, alpha=alpha, generations=50)
    operator_costs.append(dist / 1000) # Convert to km
    commuter_costs.append(ride / 1000) # Convert to pax-km

# Plotting the CTSEM Hero Image
plt.figure(figsize=(10, 6))
plt.plot(operator_costs, commuter_costs, marker='o', linestyle='-', color='b', markersize=8)
plt.title("Pareto Front: Operator Cost vs. Commuter Ride Time (Andheri Hub)", fontsize=14)
plt.xlabel("Total System Mileage (Operator Cost in km)", fontsize=12)
plt.ylabel("Total Passenger Ride Time (Commuter Cost in pax-km)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add annotations to explain the curve to reviewers
plt.annotate('Selfish Algorithm\n(Saves Gas, Ignores Pax)', xy=(operator_costs[0], commuter_costs[0]), 
             xytext=(operator_costs[0]+2, commuter_costs[0]+5), arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate('Generous Algorithm\n(Fastest Rides, Spends Gas)', xy=(operator_costs[-1], commuter_costs[-1]), 
             xytext=(operator_costs[-1]-5, commuter_costs[-1]-10), arrowprops=dict(facecolor='green', shrink=0.05))

plt.savefig("pareto_front_ctsem.png", dpi=300, bbox_inches='tight')
print("Saved Pareto Front image as pareto_front_ctsem.png!")