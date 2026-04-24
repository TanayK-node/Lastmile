import pandas as pd
import numpy as np
import folium
import networkx as nx
from sklearn.cluster import DBSCAN
import os

import osmnx_router as oxr
import route_clusterer as rc
import tsp_sequencer as tsp
import ctsem_evaluator as ce

def generate_virtual_stops(demand_df):
    if len(demand_df) < 5: 
        return None
    coords = np.radians(demand_df[['lat', 'lon']])
    epsilon = (300 / 1000.0) / 6371.0088 
    db = DBSCAN(eps=epsilon, min_samples=4, algorithm='ball_tree', metric='haversine').fit(coords)
    demand_df = demand_df.copy()
    demand_df['hub_cluster_id'] = db.labels_

    valid_clusters = demand_df[demand_df['hub_cluster_id'] != -1]
    virtual_stops = []
    
    stop_idx = 1
    for hub_id, hub_data in valid_clusters.groupby('hub_cluster_id'):
        virtual_stops.append({
            'matrix_idx': stop_idx, 'stop_id': f"Hub_{hub_id}",
            'lat': hub_data['lat'].mean(), 'lon': hub_data['lon'].mean(),
            'unique_commuters': hub_data['device_aid'].nunique()
        })
        stop_idx += 1

    if not virtual_stops:
        return None

    return pd.DataFrame(virtual_stops).sort_values(by='unique_commuters', ascending=False)

def get_street_path(G, orig_node, dest_node):
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight='travel_time')
        return [[G.nodes[n]['y'], G.nodes[n]['x']] for n in route]
    except nx.NetworkXNoPath:
        return []

def draw_final_map(G, all_routes, depot_node, station_name, scenario_name, output_filename):
    depot_lat, depot_lon = G.nodes[depot_node]['y'], G.nodes[depot_node]['x']
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=14, tiles='CartoDB positron')
    
    folium.Marker([depot_lat, depot_lon], popup=f"<b>{station_name} Depot</b>", icon=folium.Icon(color='black', icon='train', prefix='fa')).add_to(m)
    route_colors = ['#e6194b', '#4363d8', '#3cb44b', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

    for idx, route in enumerate(all_routes):
        color = route_colors[idx % len(route_colors)]
        bus_name = f"{scenario_name} Route {idx + 1}"
        full_path_coords = []
        
        if len(route) > 0: full_path_coords.extend(get_street_path(G, depot_node, route[0]['osmnx_node']))
        for i in range(len(route) - 1): full_path_coords.extend(get_street_path(G, route[i]['osmnx_node'], route[i+1]['osmnx_node']))
        if len(route) > 0: full_path_coords.extend(get_street_path(G, route[-1]['osmnx_node'], depot_node))

        for stop in route:
            folium.CircleMarker(location=[stop['lat'], stop['lon']], radius=7, color=color, fill=True, fill_opacity=0.9, tooltip=f"<b>{bus_name}</b><br>Demand: {stop['unique_commuters']}").add_to(m)
        if full_path_coords:
            folium.PolyLine(locations=full_path_coords, color=color, weight=5, opacity=0.8, tooltip=bus_name).add_to(m)

    m.save(output_filename)

if __name__ == "__main__":
    print("==========================================")
    print("INITIALIZING FULL CITY-WIDE 4D AI ROUTING ENGINE")
    print("==========================================\n")

    STATIONS = {
        "Andheri": (19.1197, 72.8464),
        "Bandra": (19.0544, 72.8402),
        "Borivali": (19.2291, 72.8573),
        "Goregaon": (19.1645, 72.8495),
        "Churchgate": (18.9322, 72.8264)
    }
    
    print("Loading 4D Temporal Demand Matrices...")
    try:
        dfs = {
            "AM_Feeder": pd.read_csv("./demand_matrices/AM_feeder_demand.csv"),
            "AM_Dispersal": pd.read_csv("./demand_matrices/AM_dispersal_demand.csv"),
            "PM_Feeder": pd.read_csv("./demand_matrices/PM_feeder_demand.csv"),
            "PM_Dispersal": pd.read_csv("./demand_matrices/PM_dispersal_demand.csv")
        }
    except FileNotFoundError:
        print("Error: Demand CSVs not found. Run chain_extractor.py!")
        exit()

    output_dir = "./final_fleet_maps"
    os.makedirs(output_dir, exist_ok=True)

    # City-wide Accumulators for the final evaluation
    city_times, city_distances, city_virtual_stops, city_demand_dfs = [], [], [], []

    for station_name, (s_lat, s_lon) in STATIONS.items():
        print(f"\n==========================================")
        print(f"PROCESSING STATION: {station_name}")
        print(f"==========================================")
        
        # Dynamically load this station's map (Fast!)
        G = oxr.load_or_download_station_graph(station_name, s_lat, s_lon)

        for scenario_name, scenario_df in dfs.items():
            print(f"\n  --- {scenario_name} ---")
            station_demand = scenario_df[scenario_df['station'] == station_name]
            
            if station_demand.empty:
                print("      [-] No demand data for this scenario.")
                continue
                
            city_demand_dfs.append(station_demand)
            virtual_stops = generate_virtual_stops(station_demand)
            
            if virtual_stops is not None and len(virtual_stops) > 0:
                virtual_stops_dict = virtual_stops.to_dict('records')
                
                # PHASE A: Matrices + Snapping
                dist_matrix, time_matrix, depot_node = oxr.build_osmnx_distance_matrix(G, s_lat, s_lon, virtual_stops_dict)
                virtual_stops = pd.DataFrame(virtual_stops_dict)
                city_virtual_stops.append(virtual_stops)
                
                # PHASE B: Stage-1 Genetic Clusterer
                zoned_stops_df = rc.cluster_hubs_into_routes(virtual_stops, time_matrix)
                
                station_routes = []
                # PHASE C: Stage-2 TSP Sequencer
                for zone_id, zone_data in zoned_stops_df.groupby('route_zone_id'):
                    ordered_route, r_time, r_dist = tsp.optimize_cluster_route(zone_data.to_dict('records'), time_matrix, dist_matrix, depot_idx=0, pop_size=50, generations=50)
                    station_routes.append(ordered_route)
                    city_times.append(r_time)
                    city_distances.append(r_dist)
                
                # Mapping
                map_name = os.path.join(output_dir, f"{station_name}_{scenario_name}_Route.html")
                draw_final_map(G, station_routes, depot_node, station_name, scenario_name, map_name)
                print(f"      [+] Dispatched {len(station_routes)} Routes. Map Saved.")
            else:
                print("      [-] Insufficient density for AI routing.")
            
    # PHASE D: Grand Evaluation
    print("\n==========================================")
    print("CITY-WIDE CTSEM FINAL EVALUATION METRICS")
    print("==========================================")
    
    combined_demand = pd.concat(city_demand_dfs, ignore_index=True) if city_demand_dfs else pd.DataFrame()
    
    eval_df = ce.evaluate_route_geometry(
        combined_demand, 
        city_virtual_stops, 
        city_times, 
        city_distances
    )
    print("\n" + eval_df.to_markdown(index=False) + "\n")