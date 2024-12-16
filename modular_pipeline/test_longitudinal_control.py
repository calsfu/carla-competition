from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
from lateral_control import LateralController
from longitudinal_control import LongitudinalController
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import cv2

# action variables
speed = 0
a = np.zeros(3)
# init carla environement

# define variables
steps = 0

# init modules of the pipeline
LD_module = LaneDetection()
LatC_module = LateralController()
LongC_module = LongitudinalController()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

# while True:
for i in range(10):
    # perform step
    s = cv2.imread('0017.png')
    # lane detection
    lane1, lane2 = LD_module.lane_detection(s)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2)
    target_speed = target_speed_prediction(waypoints, max_speed=60, exp_constant=4.5)

    # control
    a[0] = LatC_module.stanley(waypoints, speed)
    a[1], a[2] = LongC_module.control(speed, target_speed)

    # output and plot figure
    
    print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
    print("speed {:+0.2f} targetspeed {:+0.2f}".format(speed, target_speed))

    #LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
    LongC_module.plot_speed(speed, target_speed, steps, fig)
    steps += 1
    # check if stop
