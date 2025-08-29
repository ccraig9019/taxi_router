import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

#Read CSV file
data = pd.read_csv("data.csv")

#Optional: display the data
print("Test data loaded:")
print(data)

#Extract relevant columns
addresses = data["Address"].tolist()
num_students = data["Number of Students"].tolist()

print("\nAddresses:", addresses)
print("Number of students per address:", num_students)

#Defining fake travel times (in minutes)
#Airport is node 0, family 1 is node 1, family 2 is node 2, and so on
#Number of nodes = n+1, where n=number of families (because of the airport)

time_matrix = [
    [0, 10, 15, 20, 25, 30],   # airport -> families
    [10, 0, 5, 15, 20, 25],    # fam1 -> others
    [15, 5, 0, 10, 15, 20],    # fam2 -> others
    [20, 15, 10, 0, 5, 10],    # fam3 -> others
    [25, 20, 15, 5, 0, 5],     # fam4 -> others
    [30, 25, 20, 10, 5, 0],    # fam5 -> others
]

num_families = len(num_students)
num_taxis = 10 #placeholder value
DEPOT = 0 #airport
SINK = num_families+1 #optional endpoint for taxi routes, a node that is 0 minutes away from any given node, so that taxis can end their journey after the last dropoff


#Routing model setup
manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_taxis, DEPOT)
routing = pywrapcp.RoutingModel(manager)

# Transit callback (travel time)
def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return time_matrix[from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(time_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

#Student capacity dimension (max 6 per taxi)
def demand_callback(from_index):
    node = manager.IndexToNode(from_index)
    if node == 0:
        return 0
    return num_students[node-1]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    [15]*num_taxis,
    True,
    "StudentLoad"
)

#Max stops dimension (max 2 stops per taxi)
def stops_callback(from_index):
    node = manager.IndexToNode(from_index)
    if node == 0:
        return 0
    return 1

stops_callback_index = routing.RegisterUnaryTransitCallback(stops_callback)
routing.AddDimensionWithVehicleCapacity(
    stops_callback_index,
    0,
    [3]*num_taxis,
    True,
    "StopCount"
)

#Search parameters
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

#Solve the problem
solution = routing.SolveWithParameters(search_parameters)

#Display solution
if solution:
    taxi_number = 1
    for vehicle_id in range(num_taxis):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for taxi {vehicle_id + 1}:\nAirport ->"
        route_students = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                plan_output += f"{addresses[node-1]} ({num_students[node-1]} students) -> "
                route_students += num_students[node-1]
            index = solution.Value(routing.NextVar(index))

        #Skip printing any empty taxis
        if route_students == 0:
            continue

        plan_output = plan_output.rstrip(" -> ")
        plan_output += "\n"
        plan_output += f"Total students: {route_students}\n"
        print(plan_output)
else:
    print("No solution found!")