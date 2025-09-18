import os
import json
import math
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


#Load API key
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("No API_KEY found in environment. Put your key in a .env file as API_KEY=...")

#Read CSV file
data = pd.read_csv("data.csv")
addresses = data["Address"].astype(str).tolist()
num_students = data["Number of Students"].astype(int).tolist()

#Calling specific addresses as locations and appending "Edinburgh, UK" to avoid mismatching in Google Maps
locations = ["Edinburgh Airport"] + [addr + ", Edinburgh, UK" for addr in addresses]

#Distance matrix builder with caching and batching
#Cache filename to avoid repeated API calls during development
CACHE_FILE = "distance_matrix_cache.json"

def load_cached_matrix(cache_file, locations):
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)
    #validate that cached locations match (simple string equality)
    if cache.get("locations") == locations:
        return cache.get("matrix")
    return None

def save_cached_matrix(cache_file, locations, matrix):
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"locations": locations, "matrix": matrix}, f)

def build_distance_matrix(locations, api_key, verbose=True):
    """
    Build a full square distance matrix (meters) for 'locations' using Google Distance Matrix API, batching as required.

    Limits handled:
        - max 25 origins or 25 destinations per request
        - max elements per request typically 100 (origins * destinations <= 100)
    """
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    n = len(locations)
    #initialise with very large numbers (penalty) so infeasible arcs are not chosen
    INF = 10**9
    matrix = [[INF] * n for _ in range(n)]

    #chunk destinations into blocks up to 25 each
    max_per_dim = 25 #to respect Google API limits - no more than 25 origins or destinations per request
    dest_chunks = [locations[i:i+max_per_dim] for i in range(0, n, max_per_dim)]
    for d_chunk_idx, dest_chunk in enumerate(dest_chunks):
        dest_chunk_size = len(dest_chunk)
        dest_str = "|".join(dest_chunk)
        #compute how many origins we can put per request such that origins * dest_chunk_size <= 100 and <= 25
        max_orig_per_req = min(max_per_dim, max(1, 100 // dest_chunk_size)) #to respect Google API limits - no more than 100 total elements (so origins x destinations)
        #now chunk origins accordingly
        origin_chunks = [locations[i:i+max_orig_per_req] for i in range(0, n, max_orig_per_req)]

        for o_chunk_idx, origin_chunk in enumerate(origin_chunks):
            orig_str = "|".join(origin_chunk)
            params = {
                "origins": orig_str,
                "destinations": dest_str,
                "mode": "driving",
                "key": api_key
            }

            if verbose:
                orig_start = o_chunk_idx * max_orig_per_req
                dest_start = d_chunk_idx * dest_chunk_size
                print(f"Requesting origins [{orig_start}:{orig_start+len(origin_chunk)}] -> "
                      f"destinations [{dest_start}:{dest_start+dest_chunk_size}] ...", end=" ")
                
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP error from Distance Matrix API: {resp.status_code} {resp.text}")
            data = resp.json()
            if data.get("status") != "OK":
                raise RuntimeError(f"Distance Matrix API returned status={data.get('status')}; message={data.get('error_message')}")
            
            rows = data.get("rows", [])
            #fill matrix for this chunk
            for i_local, row in enumerate(rows):
                for j_local, element in enumerate(row.get("elements", [])):
                    global_i = o_chunk_idx * max_orig_per_req + i_local
                    global_j = d_chunk_idx * dest_chunk_size + j_local
                    if element.get("status") == "OK":
                        matrix[global_i][global_j] = int(element["distance"]["value"])
                    else:
                        matrix[global_i][global_j] = INF
            if verbose:
                print("done.")
            #small sleep to be polite (and avoid hitting QPS limits)
            time.sleep(0.1)
    return matrix

#Try loading cached matrix first
distance_matrix = load_cached_matrix(CACHE_FILE, locations)
if distance_matrix is None:
    print("No cached distance matrix found or locations changed. Building new matrix (this may take a moment)...")
    distance_matrix = build_distance_matrix(locations, API_KEY, verbose=True)
    #save in cach (matrix is integers; JSON can hold it)
    save_cached_matrix(CACHE_FILE, locations, distance_matrix)
else:
    print("Loaded distance matrix from cache.")

#print small preview for debug
print("\nSample distance matrix (meters) preview:")
for row in distance_matrix[:min(6, len(distance_matrix))]:
    print(row[:min(6, len(row))])


#----OR-Tools model setup (using distances in meters as cost)------#
n_locations = len(locations)
num_families = len(num_students)
#choose maximum vehicles allowed (upper bound). Use one taxi per family as upper bound:
num_taxis = max(1, len(addresses))
DEPOT = 0

#create manager and routing model
manager = pywrapcp.RoutingIndexManager(n_locations, num_taxis, DEPOT)
routing = pywrapcp.RoutingModel(manager)

# Transit (distance) callback
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    #OR-Tools expects integers
    return int(distance_matrix[from_node][to_node])

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

#Capacity constraints (students)
vehicle_capacity = 6
def demand_callback(from_index):
    node = manager.IndexToNode(from_index)
    if node == DEPOT:
        return 0
    return int(num_students[node-1])

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    [vehicle_capacity]*num_taxis,
    True,
    "StudentLoad"
)

#Stop count constraint (max stops per taxi)
max_stops_per_taxi = 3
def stops_callback(from_index):
    node = manager.IndexToNode(from_index)
    if node == DEPOT:
        return 0
    return 1

stops_callback_index = routing.RegisterUnaryTransitCallback(stops_callback)
routing.AddDimensionWithVehicleCapacity(
    stops_callback_index,
    0,
    [max_stops_per_taxi] * num_taxis,
    True,
    "StopCount"
)

#Optional: discourage using many taxis with a fixed cost (comment out if you prefer)
#The unit here is the same as arc costs (meters). Choose a value large enough to make opening a taxi non-trivial.
#routing.SetFixedCostOfAllVehicles(100000) #e.g.: 100 km penalty to open a new taxi


#Search parameters
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
search_parameters.time_limit.FromSeconds(10) #adjust if you want a longer search

#Solve the problem
solution = routing.SolveWithParameters(search_parameters)

#----Extract and display solution ----#
if solution:
    taxi_number = 1
    total_overall_distance = 0
    for vehicle_id in range(num_taxis):
        index = routing.Start(vehicle_id)
        route_students = 0
        route_addresses = []
        route_distance = 0 #meters
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            #record family stops (skip depot)
            if node != DEPOT:
                route_addresses.append((locations[node], num_students[node-1]))
                route_students += num_students[node-1]
            #add arc distance (including from depot -> first stop and between stops)
            route_distance += distance_matrix[node][next_node]
            index = next_index

        #Skip printing any empty taxis
        if route_students == 0:
            continue

        #print route
        print(f"\nRoute for taxi {taxi_number}:")
        #show as "Airport -> addr (n students) -> addr2..."
        route_str = "Airport"
        for addr, count in route_addresses:
            route_str += f" -> {addr} ({count} students)"
        print(route_str)
        print(f"Total students: {route_students}")
        print(f"Total distance: {route_distance/1000:.2f} km")
        total_overall_distance += route_distance
        taxi_number += 1

    print(f"\nOverall distance across used taxis: {total_overall_distance/1000:.2f} km")

else:
    print("No solution found!")
