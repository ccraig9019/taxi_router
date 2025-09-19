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

# Read CSV file
data = pd.read_csv("data.csv")

# Check for missing or invalid values in 'Number of Students'
if data["Number of Students"].isnull().any():
    bad_rows = data[data["Number of Students"].isnull()]
    raise ValueError(
        f"Missing student counts in these rows:\n{bad_rows[['Address','Number of Students']]}"
    )

# Convert safely to int
try:
    num_students = data["Number of Students"].astype(int).tolist()
except ValueError as e:
    raise ValueError(
        "Error converting 'Number of Students' to integers. "
        "Check your CSV for non-numeric values."
    ) from e

# Addresses
addresses = data["Address"].astype(str).tolist()

#Calling specific addresses as locations and appending "Edinburgh, UK" to avoid mismatching in Google Maps
locations = ["Edinburgh Airport, Edinburgh, UK"] + [addr + ", Edinburgh, UK" for addr in addresses]

#Distance matrix builder with caching and batching
#Cache filename to avoid repeated API calls during development
CACHE_FILE = "distance_matrix_cache.json"

#global penalty for unreachable arcs (meters). Used for debugging and to avoid missing entries
INF = 10**9

def load_cached_matrix(cache_file, locations):
    """Return (matrix, origins, destinations) or None."""
    if not os.path.exists(cache_file):
        return None
    with open(cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)
    #validate that cached locations match (simple string equality)
    if cache.get("locations") == locations:
        return cache.get("matrix"), cache.get("origins"), cache.get("destinations")
    return None

def save_cached_matrix(cache_file, locations, matrix, origins, destinations):
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({
            "locations": locations,
            "matrix": matrix,
            "origins": origins,
            "destinations": destinations
        }, f)

def build_distance_matrix(locations, api_key, verbose=True):
    """
    Build a full square distance matrix (meters) for 'locations' using Google Distance Matrix API, batching as required.
    Returns: matrix, resolved_origins, resolved_destinations
    """
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    n = len(locations)

    origin_indices = list(range(n))
    dest_indices = list(range(n))


    # initialize matrix with INF
    matrix = [[INF] * n for _ in range(n)]

    # Preallocate lists to hold resolved addresses for each global index
    all_origin_addresses = [None] * n
    all_destination_addresses = [None] * n

    # chunking parameters
    max_per_dim = 25
    dest_chunks = [dest_indices[i:i+max_per_dim] for i in range(0, n, max_per_dim)]

    for d_chunk_idx, dest_chunk in enumerate(dest_chunks):
        dest_chunk_size = len(dest_chunk)
        dest_chunk_addrs = [locations[j] for j in dest_chunk]
        dest_str = "|".join(dest_chunk_addrs)


        # choose number of origins per request so origins * dest_chunk_size <= 100
        max_orig_per_req = min(max_per_dim, max(1, 100 // dest_chunk_size))
        origin_chunks = [origin_indices[i:i+max_orig_per_req] for i in range(0, n, max_orig_per_req)]


        for o_chunk_idx, origin_chunk in enumerate(origin_chunks):
            orig_addrs = [locations[i] for i in origin_chunk]
            orig_str = "|".join(orig_addrs)

            
            params = {
                "origins": orig_str,
                "destinations": dest_str,
                "mode": "driving",
                "key": api_key
            }

            if verbose:
                orig_start = origin_chunk[0]
                dest_start = dest_chunk[0]
                print(f"Requesting origins [{orig_start}:{orig_start+len(origin_chunk)}] -> "
                      f"destinations [{dest_start}:{dest_start+dest_chunk_size}] ...", end=" ")

            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP error from Distance Matrix API: {resp.status_code} {resp.text}")
            resp_json = resp.json()
            if resp_json.get("status") != "OK":
                raise RuntimeError(f"Distance Matrix API returned status={resp_json.get('status')}; message={resp_json.get('error_message')}")

            # Record resolved addresses into their global positions
            resp_origins = resp_json.get("origin_addresses", [])
            resp_dests = resp_json.get("destination_addresses", [])
            for i_local, addr in enumerate(resp_origins):
                global_i = origin_chunks[o_chunk_idx][i_local]
                all_origin_addresses[global_i] = addr
            for j_local, addr in enumerate(resp_dests):
                global_j = dest_chunk[j_local]
                all_destination_addresses[global_j] = addr


            # Fill matrix cells for this chunk and log INF insertions
            rows = resp_json.get("rows", [])
            for i_local, row in enumerate(rows):
                for j_local, element in enumerate(row.get("elements", [])):
                    global_i = origin_chunks[o_chunk_idx][i_local]
                    global_j = dest_chunk[j_local]
                    if element.get("status") == "OK":
                        matrix[global_i][global_j] = int(element["distance"]["value"])
                    else:
                        matrix[global_i][global_j] = INF
                        # log a clear, index-aligned warning
                        print(
                            f"WARNING: INF inserted for origin[{global_i}] -> destination[{global_j}]: "
                            f"origin='{locations[global_i]}' dest='{locations[global_j]}' "
                            f"(API status={element.get('status')})"
                        )

            if verbose:
                print("done.")
            time.sleep(0.1)

    # If any resolved address is still None, leave it None — we'll print them for debugging later
    return matrix, all_origin_addresses, all_destination_addresses

#Try loading cached matrix first
cached = load_cached_matrix(CACHE_FILE, locations)
if cached is None:
    print("No cached distance matrix found or locations changed. Building new matrix (this may take a moment)...")
    distance_matrix, resolved_origins, resolved_destinations = build_distance_matrix(locations, API_KEY, verbose=True)
    #save in cach (matrix is integers; JSON can hold it)
    save_cached_matrix(CACHE_FILE, locations, distance_matrix, resolved_origins, resolved_destinations)
else:
    distance_matrix, resolved_origins, resolved_destinations = cached
    print("Loaded distance matrix from cache.")

#print small preview for debug
print("\nSample distance matrix (meters) preview:")
for row in distance_matrix[:min(6, len(distance_matrix))]:
    print(row[:min(6, len(row))])

#print what Google resolved
print("\nResolved origins (from API):")
for i, addr in enumerate(resolved_origins):
    print(f"{i}: {addr}")

print("\nResolved destinations (from API):")
for i, addr in enumerate(resolved_destinations):
    print(f"{i}: {addr}")

#quick validation check: warn if resolved address does not include 'Edinburgh' or 'UK'
print("\nValidation warnings (if any):")
for i, addr in enumerate(resolved_destinations):
    if addr and ("Edinburgh" not in addr and "UK" not in addr):
        print(f"   >>> Warning: location {i} resolved to '{addr}', which does not contain 'Edinburgh', or 'UK'. Input was: {locations[i]}")


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
search_parameters.time_limit.FromSeconds(30) #adjust if you want a longer search

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
        route_distance = 0  # meters
        route_has_INF = False

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)

            if node != DEPOT:
                route_addresses.append((locations[node], num_students[node-1]))
                route_students += num_students[node-1]

            arc_dist = distance_matrix[node][next_node]
            if arc_dist >= INF:
                print(f"ERROR: route includes unreachable arc origin[{node}] -> dest[{next_node}]: "
                      f"'{locations[node]}' -> '{locations[next_node]}' (INF used).")
                route_has_INF = True
                # don't add INF numerically to the distance (it will skew totals). mark route invalid instead.
            else:
                route_distance += arc_dist

            index = next_index

        if route_has_INF:
            print(f"\nRoute for taxi {taxi_number} contains unreachable leg(s). Check the address resolution above.")
            # You can choose to continue (print partial route) or skip printing this route:
            # continue

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
