import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LongitudinalController:
    '''
    Longitudinal Control using a PID Controller

    functions:
        PID_step()
        control()
    '''
    def __init__(self, KP=0.01, KI=0.0, KD=0.0):
        self.last_error = 0
        self.sum_error = 0
        self.last_control = 0
        self.speed_history = []
        self.target_speed_history = []
        self.step_history = [] 

        # PID parameters
        self.KP = KP
        self.KI = KI
        self.KD = KD

    def PID_step(self, speed, target_speed):
        '''
        ##### TODO ####
        Perform one step of the PID control
        - Implement the descretized control law.
        - Implement a maximum value for the sum of error you are using for the intgral term 

        args: 
            speed
            target_speed

        output: 
            control (u)
        '''
        
        # define error from set point target_speed to speed 
        error = target_speed - speed
        max_integral_error = 100
        # derive PID elements
        P = self.KP * error
        I = self.KI * self.sum_error
        D = self.KD * (error - self.last_error)

        control = P + I + D
        
        # update previous values
        self.last_error = error
        self.sum_error += error
        if self.sum_error > max_integral_error:
            self.sum_error = max_integral_error
        elif self.sum_error < -max_integral_error:
            self.sum_error = -max_integral_error

        self.last_control = control

        return control

    def control(self, speed, target_speed):
        '''
        Derive action values for gas and brake via the control signal
        using PID controlling

        Args:
            speed (float)
            target_speed (float)

        output:
            gas
            brake
        '''

        control = self.PID_step(speed, target_speed)
        brake = 0
        gas = 0

        # translate the signal from the PID controller 
        # to the action variables gas and brake
        if control >= 0:
            gas = np.clip(control, 0, 0.8) 
        else:
            brake = np.clip(-1*control, 0, 0.8)

        return gas, brake

    def plot_speed(self, speed, target_speed, step, fig):
        self.speed_history.append(speed)
        self.target_speed_history.append(target_speed)
        self.step_history.append(step)

        if step == 9:
            plt.gcf().clear()
            plt.plot(self.step_history, self.speed_history, c="green")
            plt.plot(self.step_history, self.target_speed_history)
            plt.pause(10)
            plt.legend(['speed', 'target_speed'])
            
            fig.canvas.flush_events()