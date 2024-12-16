import random
import numpy as np
import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

# The reward function you provided.
def reward_function(collision, speed, lane_invasion, current_position, next_waypoint, previous_distance, target_speed=15.0):
    """
    Calculate the reward for the agent based on its actions.

    Parameters:
    - collision (bool): Whether a collision occurred.
    - speed (float): The current speed of the vehicle.
    - lane_invasion (bool): Whether the vehicle has invaded a lane.
    - current_position (tuple): The current (x, y) position of the vehicle.
    - next_waypoint (tuple): The (x, y) position of the next waypoint.
    - target_speed (float): The target speed for the vehicle.

    Returns:
    - float: The calculated reward.
    """
    
    # Penalty for collisions.
    if collision:
        return -100.0

    # Penalty for lane invasions.
    if lane_invasion:
        return -10.0

    # Reward for maintaining speed.
    speed_reward = 1.0 - abs(speed - target_speed) / target_speed
    speed_reward = max(speed_reward, 0)  # Ensure non-negative reward.

    # Calculate Euclidean distance to next waypoint.
    distance_to_waypoint = np.linalg.norm(np.array(current_position) - np.array(next_waypoint))

    # Positive reward for getting closer to the next waypoint.
    goal_reward = max(0, 1 - distance_to_waypoint)  # Reward increases as distance decreases.
    # Negative reward for moving away from the next waypoint.
    if distance_to_waypoint > previous_distance:  # We are assuming you keep track of the previous distance.
        distance_penalty = -1  # You can adjust this value.
    else:
        distance_penalty = 0

    # Combine rewards.
    total_reward = (speed_reward * 100.0) + goal_reward + distance_penalty

    return total_reward

# The generate_route function from your code (assume it is already defined).
def generate_route(start_index, end_index, map_name='Town01', sampling_resolution=2):
    """
    -Generate a route between two points in the Carla simulator.
    -This code give you the waypoints for the reward function.
    -map_name=Change this according to your choice.
    """
    # Connect to Carla client and load the world.
    client = carla.Client("localhost", 2000)
    client.set_timeout(10)
    world = client.load_world(map_name)
    amap = world.get_map()

    # Set up the GlobalRoutePlanner.
    dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()

    # Get the spawn points and select the start and end locations based on indices.
    spawn_points = world.get_map().get_spawn_points()
    start_location = carla.Location(spawn_points[start_index].location)
    end_location = carla.Location(spawn_points[end_index].location)

    # Generate the route between the two locations.
    route = grp.trace_route(start_location, end_location)

    return route

# Test the reward function.
# Not intended for use just for testing.
def test_reward_function():

    ##THIS IS TEST FUNCTION NOT INTENDED FOR REAL USE.
    #IF YOU WANT TO USE THIS FUNCTION CHANGE THE BELOW PARAMETERS FOR YOUR CASE.
    # Pick a start and end index for waypoints.
    start_index = random.randint(0, 200)  # Choose a start waypoint index.
    end_index = random.randint(0, 200)  # Choose an end waypoint index.

    # Generate the route using the generate_route function.
    route = generate_route(start_index, end_index)
    
    # Iterate through waypoints in the route and calculate reward for each transition.
    total_reward = 0
    for i in range(1, len(route)):  # Start from the second waypoint.
        # Get the current and next waypoint locations.
        collision = random.choice([True, False])  # Randomly simulate collision.
        speed = random.uniform(0, 30)  # Random vehicle speed between 0 and 30.
        lane_invasion = random.choice([True, False]) 
        current_waypoint = route[i-1][0].transform.location  # Current waypoint.
        next_waypoint = route[i][0].transform.location  # Next waypoint.
        current_position = (random.uniform(-100, 100), random.uniform(-100, 100))  # Random vehicle position.

        # Calculate the reward for the transition between current and next waypoint.
        reward = reward_function(collision, speed, lane_invasion, current_position, (next_waypoint.x, next_waypoint.y), target_speed=15.0)

        # Output the result.
        print(f"Current Position: {current_position}")
        print(f"Next Waypoint: {(next_waypoint.x, next_waypoint.y)}")
        print(f"Speed: {speed:.2f}, Collision: {collision}, Lane Invasion: {lane_invasion}")
        print(f"Calculated Reward: {reward:.2f}")
        total_reward += reward

    print(f"Total Reward for the entire route: {total_reward:.2f}")
    
# test_reward_function()