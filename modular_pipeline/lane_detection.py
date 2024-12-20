import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import cv2
import torch
from adscnet import ADSCNet

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=120) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=120, spline_smoothness=15, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([160,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size    
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0
    

    def front2bev(self, front_view_image):
        '''
        ##### TODO #####
        This function should transform the front view image to bird-eye-view image.

        input:
            front_view_image)320x240x3

        output:
            bev_image 320x240x3

        '''

        image = cv2.resize(front_view_image, (320, 240))

        src_points = np.float32([
            [0, 140],   # top-left
            [320, 140],   # top-right
            [80, 240],   # bottom-left
            [240, 240]  # bottom-right
        ])
        
        dst_points = np.float32([
            [60, 0],   # top-left
            [260, 0],   # top-right
            [140, 240],   # bottom-left
            [180, 240]  # bottom-right
        ])

        # src_points = np.float32([
        #     [0, 140],   # top-left
        #     [320, 140],   # top-right
        #     [80, 240],   # bottom-left
        #     [240, 240]  # bottom-right
        # ])
        
        # dst_points = np.float32([
        #     [0, 0],   # top-left
        #     [320, 0],   # top-right
        #     [140, 240],   # bottom-left
        #     [180, 240]  # bottom-right
        # ])

        matrix = cv2.getPerspectiveTransform(src_points,
                                            dst_points)
        bev_image = cv2.warpPerspective(image, matrix, (320, 240))

        # segmentation 
        model = ADSCNet()
        weights = torch.load('adscnet.pth', weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(weights, strict=False)
        model.eval()
        #256 x 256
        image = cv2.resize(front_view_image, (256, 256))
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        # predict
        with torch.no_grad():
            output = model(image)
        output = output.squeeze().numpy()
        output = cv2.resize(output, (320, 240))
        cv2.imshow('image', output)
        cv2.waitKey(0)

        return bev_image 


    def cut_gray(self, state_image_full):
        '''
        ##### TODO #####
        This function should cut the image at the front end of the car
        and translate to grey scale

        input:
            state_image_full 320x240x3

        output:
            gray_state_image 320x120x1

        '''
        cv2.imshow('image', state_image_full)
        cv2.waitKey(0)
        gray_state_image = cv2.cvtColor(state_image_full, cv2.COLOR_BGR2GRAY)
        #cuto the top part of the image out
        gray_state_image = gray_state_image[self.cut_size:]
        cv2.imshow('image', gray_state_image)
        cv2.waitKey(0)
        return gray_state_image[::-1]


    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 320x120x1

        output:
            gradient_sum 320x120x1

        '''
        gradient_sum = np.zeros_like(gray_image)
        gray_image = gray_image.astype(np.int16)
        for i in range(gray_image.shape[0]):
            gradient_sum[i, 120:181] = np.abs(gray_image[i, 119:180] - gray_image[i, 121:182])
            gradient_sum[i, 120:181] = np.where(gradient_sum[i, 120:181] < self.gradient_threshold, 0, gradient_sum[i, 120:181])

        assert(gradient_sum.shape == gray_image.shape)
        cv2.imshow('image', gradient_sum)
        cv2.waitKey(0)
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 320x120x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        argmaxima = np.empty((2,0))
        print(gradient_sum.shape)
        for i in range(gradient_sum.shape[0]):
            peaks = find_peaks(gradient_sum[i], distance=self.distance_maxima_gradient)[0]
            for peak in peaks:
                col = np.array([[peak], [i]])
                argmaxima = np.hstack((argmaxima, col))

        return argmaxima


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 320x120x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''
        
        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:
            
            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row],distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 160:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else: 
                    lane_boundary2_startpoint = np.array([[320,  row]])

                lanes_found = True
            
            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [320, 240, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''
        # to bev
        bev_image = self.front2bev(state_image_full)
        # to gray
        gray_state = self.cut_gray(bev_image)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)
        cv2.imshow('image', gradient_sum)
        cv2.waitKey(0)
        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)
        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)
            
            # delete first lane_boundary1_points and lane_boundary2_points from maxima
            maxima = np.delete(maxima, np.argwhere(maxima[1] == lane_boundary1_points[0][1]), axis=1)
            maxima = np.delete(maxima, np.argwhere(maxima[1] == lane_boundary2_points[0][1]), axis=1)

            while maxima.shape[1] > 0:
                # find closest maxima to the last lane boundary point
                # maxima is 2xN, lane_boundary1_points[-1] is 2x1
                distances1 = np.zeros(maxima.shape[1])
                distances2 = np.zeros(maxima.shape[1])
                
                for i in range(maxima.shape[1]):
                    distances1[i] = np.linalg.norm(maxima[:, i] - lane_boundary1_points[-1])
                    distances2[i] = np.linalg.norm(maxima[:, i] - lane_boundary2_points[-1])

                # stop loop if there is no maximum left or if the distance to the next one is too big (>=100)
                if np.min(distances1) >= 100 and np.min(distances2) >= 100:
                    break
                
                # append maxium to lane_boundary1_points or lane_boundary2_points
                # delete maximum from maxima
                if np.min(distances1) <= np.min(distances2):
                    min_idx = np.argmin(distances1)
                    lane_boundary1_points = np.vstack((lane_boundary1_points, maxima[:, min_idx]))
                    maxima = np.delete(maxima, min_idx, axis=1)
                else:
                    min_idx = np.argmin(distances2)
                    lane_boundary2_points = np.vstack((lane_boundary2_points, maxima[:, min_idx]))
                    maxima = np.delete(maxima, min_idx, axis=1)
                    
            ################
            
            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            
            #print x,y coordinates of lane_boundary1_points
            # for i in range(lane_boundary1_points.shape[0]):
                # print(lane_boundary1_points[i, 0], lane_boundary1_points[i, 1])

            if lane_boundary1_points.shape[0] > 3:
                # Pay attention: the first lane_boundary point might occur twice
                # lane_boundary 1
                x = lane_boundary1_points[:,0]
                y = lane_boundary1_points[:,1]

                lane_boundary1, _ = splprep([x, y], s=self.spline_smoothness)
            else:
                lane_boundary1 = self.lane_boundary1_old
            
            # lane_boundary 2
            if lane_boundary2_points.shape[0] > 3:
                lane_boundary2, _ = splprep([lane_boundary2_points[:,0], lane_boundary2_points[:,1]], s=self.spline_smoothness)
            else:
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # print(lane_boundary1, lane_boundary2)
        # print(len(lane_boundary1), len(lane_boundary2))
        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)

        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.pause(2)
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1], linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1], linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1], color='white')
        plt.pause(6)
        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.pause(5)
        fig.canvas.flush_events()
