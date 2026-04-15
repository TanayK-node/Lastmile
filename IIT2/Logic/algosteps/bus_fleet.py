import pandas as pd
import numpy as np
import folium

def haversine(lat1, lon1, lat2, lon2):
    """Calculates distance in meters between two points."""
    R = 6371000  
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def generate_cvrp_routes(stops_file, station_lat, station_lon, bus_capacity=40):
    print(f"Loading Virtual Stops from {stops_file}...")
    df = pd.read_csv(stops_file)
    
    total_demand = df['unique_commuters'].sum()
    min_buses = int(np.ceil(total_demand / bus_capacity))
    
    print("\n" + "="*40)
    print("FLEET OPTIMIZATION ANALYSIS")
    print("="*40)
    print(f"Total Commuter Demand: {total_demand} people")
    print(f"Maximum Bus Capacity: {bus_capacity} seats")
    print(f"Minimum Buses Required: {min_buses} buses")
    print("="*40 + "\n")

    # Sort stops by distance from the train station (furthest first)
    df['dist_to_station'] = haversine(df['lat'].values, df['lon'].values, station_lat, station_lon)
    # Convert to a list of dictionaries so we can modify the commuter counts as buses pick them up
    unassigned_stops = df.sort_values(by='dist_to_station', ascending=False).to_dict('records')

    fleet_routes = []
    bus_id = 1

    # 2. Assign Stops to Buses with DEMAND SPLITTING
    while len(unassigned_stops) > 0:
        current_bus_load = 0
        current_route = []
        stops_to_remove = []
        
        # Look through all remaining stops
        for stop in unassigned_stops:
            available_seats = bus_capacity - current_bus_load
            
            if available_seats <= 0:
                break  # The bus is completely full, send it out!
                
            if stop['unique_commuters'] > 0:
                # Figure out how many people this bus can actually take from this stop
                people_boarding = min(available_seats, stop['unique_commuters'])
                
                # Add this pick-up event to the route
                current_route.append({
                    'stop_id': stop['stop_id'],
                    'lat': stop['lat'],
                    'lon': stop['lon'],
                    'boarded': people_boarding
                })
                
                # Update the numbers
                current_bus_load += people_boarding
                stop['unique_commuters'] -= people_boarding  # Subtract the people who got on the bus
                
            # If everyone at this stop has been picked up, mark it for removal
            if stop['unique_commuters'] == 0:
                stops_to_remove.append(stop)
                
        # Remove cleared stops from the waiting list
        for s in stops_to_remove:
            unassigned_stops.remove(s)
            
        # SAFETY CATCH: Prevent infinite loops if something goes wrong
        if current_bus_load == 0:
            print("System broke loop: No passengers could be boarded.")
            break
            
        fleet_routes.append({
            'bus_name': f"Bus Route {bus_id}",
            'passengers': current_bus_load,
            'stops': current_route
        })
        
        print(f"Dispatched {fleet_routes[-1]['bus_name']} | Passengers: {current_bus_load}/{bus_capacity} | Stops Serviced: {len(current_route)}")
        bus_id += 1

    # ---------------------------------------------------------
    # 3. Visualizing the Optimized Routes
    # ---------------------------------------------------------
    print("\nMapping AI Generated Routes...")
    m = folium.Map(location=[station_lat, station_lon], zoom_start=14, tiles='CartoDB positron')

    # Origin Station
    folium.Marker(
        [station_lat, station_lon], 
        popup="<b>Andheri Station (Bus Depot)</b>", 
        icon=folium.Icon(color='red', icon='train', prefix='fa')
    ).add_to(m)

    # Extended color palette for better route differentiation
    route_colors = [
        '#e6194b',  # Red
        '#4363d8',  # Blue
        '#3cb44b',  # Green
        '#f58231',  # Orange
        '#911eb4',  # Purple
        '#46f0f0',  # Cyan
        '#f032e6',  # Magenta
        '#000075',  # Dark Blue
        '#808000',  # Olive
        '#42d4f4',  # Light Blue
        '#469990',  # Teal
        '#f0a000',  # Gold
        '#e6beff',  # Lavender
        '#c91f16',  # Dark Red
        '#9a6324'   # Brown
    ]

    for idx, route in enumerate(fleet_routes):
        color = route_colors[idx % len(route_colors)]
        bus_name = route['bus_name']
        
        route_coords = [[station_lat, station_lon]]
        
        for stop_num, stop in enumerate(route['stops']):
            route_coords.append([stop['lat'], stop['lon']])
            
            folium.Marker(
                location=[stop['lat'], stop['lon']],
                popup=f"<b>{bus_name} - Stop {stop_num + 1}</b><br>{stop['stop_id']}<br>Boarded Here: {stop['boarded']} pax",
                icon=folium.Icon(color='black', icon_color=color, icon='bus', prefix='fa')
            ).add_to(m)

        folium.PolyLine(
            locations=route_coords,
            color=color, weight=4, opacity=0.8,
            tooltip=f"{bus_name} (Total Load: {route['passengers']} pax)"
        ).add_to(m)

    m.save("optimized_fleet_routes.html")
    print("Success! Final map saved as optimized_fleet_routes.html")

# ==========================================
# HOW TO USE
# ==========================================
STOPS_FILE = 'smart_virtual_stops.csv'
STATION_LAT = 19.1197
STATION_LON = 72.8464

# Try running it now!
generate_cvrp_routes(STOPS_FILE, STATION_LAT, STATION_LON, bus_capacity=40)