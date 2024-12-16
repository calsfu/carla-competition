import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    ##### TODO #####
    Curvature as the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args: 
        waypoints [2, num_waypoints] !!!!!
    '''
    waypoints = waypoints.reshape(2, -1)
    numerator = (waypoints[:,2:] - waypoints[:, 1:-1]) @ (waypoints[:,1:-1] - waypoints[:,:-2]).T
    denominator = np.sum(np.linalg.norm(waypoints[:,2:] - waypoints[:, 1:-1], axis=0) * np.linalg.norm(waypoints[:,1:-1] - waypoints[:,:-2], axis=0))
    
    curvature = np.sum(numerator / denominator)

    return curvature


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:S
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    waypoints = waypoints.reshape(2, -1)
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints)

    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    tck1 = roadside1_spline
    tck2 = roadside2_spline

    if way_type == "center":
        ##### TODO #####
     
        # create spline arguments
        param = np.linspace(0, 1, num_waypoints)

        # derive roadside points from spline
        roadside_points1 = splev(param, tck1)
        roadside_points2 = splev(param, tck2)

        # derive center between corresponding roadside points
        roadside_points1 = np.array(roadside_points1)
        roadside_points2 = np.array(roadside_points2)
        way_points = (roadside_points1 + roadside_points2) / 2
        way_points = way_points.reshape(2, -1)

        assert way_points.shape == (2, num_waypoints)
        # output way_points with shape(2 x Num_waypoints)
        return way_points
    
    elif way_type == "smooth":
        ##### TODO #####

        # create spline arguments
        param = np.linspace(0, 1, num_waypoints)

        # derive roadside points from spline
        roadside_points1 = splev(param, tck1)
        roadside_points2 = splev(param, tck2)

        # derive center between corresponding roadside points
        roadside_points1 = np.array(roadside_points1)
        roadside_points2 = np.array(roadside_points2)
        way_points_center = (roadside_points1 + roadside_points2) / 2
        
        #take from roadside_points1 and roadside_points2 the first half and the last half
        way_points_varied = np.zeros((2, num_waypoints))
        way_points_varied[:, :num_waypoints//2] = roadside_points1[:, :num_waypoints//2]
        way_points_varied[:, num_waypoints//2:] = roadside_points2[:, num_waypoints//2:]



        # optimization
        way_points = minimize(smoothing_objective, 
                      (way_points_varied), 
                      args=way_points_center)["x"]
        way_points = way_points.reshape(2, -1)
        assert way_points.shape == (2, num_waypoints)
        return way_points


def target_speed_prediction(waypoints, num_waypoints_used=5,
                            max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''
    target_speed = (max_speed - offset_speed) * np.exp(-exp_constant * np.abs(num_waypoints_used - 2 - curvature(waypoints))) + offset_speed

    return target_speed