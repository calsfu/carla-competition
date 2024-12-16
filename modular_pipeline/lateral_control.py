import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=.1, damping_constant=.9):
        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            speed (float)
        '''
        # image is 320x120, where the car is at the bottom center
        position = np.array([160, 120])
        first_point = waypoints[:, 0]
        last_point = waypoints[:, 1]

        # derive orientation error as the angle of the first path segment to the car orientation
        orientation_error = np.arctan2(last_point[1] - first_point[1], last_point[0] - first_point[0])
        # derive cross track error as distance between desired waypoint at spline parameter equal zero ot the car position
        cross_track_error = np.min(np.linalg.norm(waypoints - position.reshape(2, 1), axis=0))
        
        # derive stanley control law
        # prevent division by zero by adding as small epsilon
        steering_angle = orientation_error + np.arctan(self.gain_constant * cross_track_error / (speed + 1e-6))

        # derive damping term
        steering_angle = steering_angle - self.damping_constant * (steering_angle - self.previous_steering_angle)
        
        self.previous_steering_angle = steering_angle

        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4) / 0.4






