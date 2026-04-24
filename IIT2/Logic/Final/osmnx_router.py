import osmnx as ox
import networkx as nx
import os

# OSMnx 2.x may default to HTTPS for Overpass, which can trigger SSL
# WRONG_VERSION_NUMBER in some Windows network setups.
ox.settings.overpass_url = "http://overpass-api.de/api"


def _resolve_place_polygon(place_query, max_results=10):
    """Return first geocoder result that is a Polygon/MultiPolygon."""
    for which_result in range(1, max_results + 1):
        try:
            gdf = ox.geocode_to_gdf(place_query, which_result=which_result)
            if not gdf.empty and gdf.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}:
                return gdf.geometry.iloc[0]
        except Exception:
            continue

    raise TypeError(
        f"Nominatim did not geocode query '{place_query}' to a Polygon/MultiPolygon in top {max_results} results."
    )

import osmnx as ox
import networkx as nx
import os

# Give the server a massive timeout window just in case
ox.settings.timeout = 1800

def load_or_download_station_graph(station_name, lat, lon):
    """
    Dynamically loads or downloads a fast, localized 7km radius 
    major-road network for the specific station being processed.
    """
    filepath = f"{station_name.lower()}_major_roads.graphml"
    
    if os.path.exists(filepath):
        print(f"  [+] Loading local Major Roads cache for {station_name}...")
        G = ox.load_graphml(filepath)
    else:
        print(f"  [!] Downloading 7km Regional Network for {station_name}... (Takes ~15s)")
        custom_filter = '["highway"~"primary|secondary|tertiary|trunk"]'
        
        G = ox.graph_from_point(
            (lat, lon), 
            dist=7000, 
            network_type='drive', 
            custom_filter=custom_filter
        )
        
        print("  [+] Calculating real-world speeds and travel times...")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        G = ox.truncate.largest_component(G, strongly=True)
        print(f"  [+] Saving graph to local cache: {filepath}")
        ox.save_graphml(G, filepath)
        
    return G

# ... (Keep your build_osmnx_distance_matrix function below this exactly the same) ...

def build_osmnx_distance_matrix(G, depot_lat, depot_lon, virtual_stops):
    """
    Builds the matrices AND snaps the Virtual Stops permanently to the Major Roads.
    """
    print("  [+] Building True Network Matrices and Snapping Stops to Major Roads...")
    nodes = []
    
    # 1. Snap Depot
    depot_node = ox.distance.nearest_nodes(G, X=depot_lon, Y=depot_lat)
    nodes.append(depot_node)
    
    # 2. Snap Stops AND update their coordinates!
    for stop in virtual_stops:
        node = ox.distance.nearest_nodes(G, X=stop['lon'], Y=stop['lat'])
        nodes.append(node)
        
        # THE FIX: Overwrite the DBSCAN coordinates with the True Road coordinates
        stop['lat'] = G.nodes[node]['y']
        stop['lon'] = G.nodes[node]['x']
        stop['osmnx_node'] = node # Save this exact intersection ID for the map
        
    n = len(nodes)
    dist_matrix = [[0] * n for _ in range(n)]
    time_matrix = [[0] * n for _ in range(n)] 
    
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    # Calculate true topological path
                    dist = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length')
                    time_sec = nx.shortest_path_length(G, nodes[i], nodes[j], weight='travel_time')
                    
                    dist_matrix[i][j] = dist
                    time_matrix[i][j] = time_sec / 60.0  
                except nx.NetworkXNoPath:
                    dist_matrix[i][j] = float('inf')
                    time_matrix[i][j] = float('inf')
                    
    print("  [+] Matrices Built & Stops Snapped Successfully.")
    return dist_matrix, time_matrix, depot_node