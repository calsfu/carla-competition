from lane_detection import LaneDetection
import matplotlib.pyplot as plt
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key
import cv2
import carla
import random 

#init carla environement
# window = pyglet.window.Window(visible=False, width=320, height=240)
# client = carla.Client('localhost', 2000)
# client.set_timeout(2.0)

# world = client.get_world()
# map = world.get_map()
# vehicle_bp = world.get_blueprint_library().find('vehicle.audi.tt')
# vehicle = world.spawn_actor(vehicle_bp, random.choice(map.get_spawn_points()), attach_to=None)

# camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
# camera_bp.set_attribute('image_size_x', '320')
# camera_bp.set_attribute('image_size_y', '240')
# camera_bp.set_attribute('fov', '90')
# camera = world.spawn_actor(camera_bp, carla.Transform(), attach_to=vehicle)

# image_data = np.zeros((240, 320, 3), dtype=np.uint8)

# def process_img(image):
#     global image_data
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (image.height, image.width, 4))
#     array = array[:, :, :3]
#     array = array[:, :, ::-1]

#     image_data = array
    
# camera.listen(process_img)
# define variables
steps = 0

# # init modules of the pipeline
LD_module = LaneDetection()

# init extra plot
fig = plt.figure()
plt.ion()
plt.show()

s = cv2.imread('0017.png')
# cv2.imshow('image', s)
# cv2.waitKey(0)

#resize to240x320
s = cv2.resize(s, (320, 240))
# cv2.imshow('image', s)
# cv2.waitKey(0)

# lane detection
splines = LD_module.lane_detection(s)

s = LD_module.front2bev(front_view_image=s)
# plot figure
if steps % 2 == 0:
    LD_module.plot_state_lane(s, steps, fig)
steps += 1

# image = pyglet.image.ImageData(320, 240, 'RGB', image_data.tobytes())

# @window.event
# def on_draw():
#     window.clear()
#     image.blit(0, 0)


# pyglet.app.run()

# while True:
#     # perform step
#     # world.tick()
#     # image = pyglet.image.ImageData(320, 240, 'RGB', image_data.tobytes())

#     # world.render(pyglet.window )
#     # s = image_data
#     # if s is None:
#     #     continue
#     # print(s.shape)

#     s = cv2.imread('0017.png')
#     # lane detection
#     splines = LD_module.lane_detection(s)
    
#     # plot figure
#     if steps % 2 == 0:
#         LD_module.plot_state_lane(s, steps, fig)
#     steps += 1

#     # check if stop

