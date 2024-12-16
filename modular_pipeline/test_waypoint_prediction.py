from lane_detection import LaneDetection
from waypoint_prediction import waypoint_prediction, target_speed_prediction
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import cv2

# init carla environement

# define variables
steps = 0

# init modules of the pipeline
LD_module = LaneDetection()

# init extra plot
fig = plt.figure()
plt.ion()

if True:
    # perform step
    s = cv2.imread('0017.png')
    # cv2.imshow('image', s)
    # cv2.waitKey(0)

    #resize to240x320
    s = cv2.resize(s, (320, 240))
    # lane detection
    lane1, lane2 = LD_module.lane_detection(s)

    # waypoint and target_speed prediction
    waypoints = waypoint_prediction(lane1, lane2, way_type='smooth')
    target_speed = target_speed_prediction(waypoints)
    print("Target Speed: ", target_speed)
    # plot figure
    s = LD_module.front2bev(front_view_image=s)
    LD_module.plot_state_lane(s, steps, fig, waypoints=waypoints)
    # check if stop
