# carla-competition

# Imitation Learning

This repository implements a pipeline for training an imitation learning model in the CARLA simulator. The goal is to collect driving data from the CARLA autopilot and train a convolutional neural network (CNN) to predict driving commands from front-camera images. This project serves as an entry point for building autonomous vehicle models based on supervised learning.

---

## Key Features
### Data Collection
- The `data_collect.py` script modifies CARLA's `manual_control.py` to enable data collection while driving in autopilot mode.
- Captures **front-camera RGB images** and **corresponding driving commands**.
- Saves the data in an organized structure, ready for training.

### Neural Network Architecture
- A custom-designed convolutional neural network (CNN) processes RGB images and predicts driving commands.
- The network features:
  - Five convolutional layers to extract spatial features from images.
  - Fully connected layers to map features to driving commands.
  - ReLU activation functions for non-linearity.

### Training Utilities
- PyTorch-based dataset and network definitions for seamless model training.
- A loss plotting script to monitor and visualize model performance during training.

---

## Neural Network Architecture

| Layer Type         | Parameters                              | Output Shape       | Description                                       |
|--------------------|-----------------------------------------|--------------------|---------------------------------------------------|
| **Input Layer**    | RGB Images (3 channels, 240x320 pixels)| (3, 240, 320)      | Input front-camera image from the CARLA simulator. |
| **Conv2D (1)**     | Filters: 24, Kernel: 5x5, Stride: 2     | (24, 118, 158)     | Extracts low-level spatial features.             |
| **ReLU Activation**| -                                       | (24, 118, 158)     | Non-linear activation for better feature learning.|
| **Conv2D (2)**     | Filters: 36, Kernel: 5x5, Stride: 2     | (36, 57, 77)       | Extracts mid-level spatial features.             |
| **ReLU Activation**| -                                       | (36, 57, 77)       |                                                   |
| **Conv2D (3)**     | Filters: 48, Kernel: 5x5, Stride: 2     | (48, 27, 37)       | Learns high-level spatial patterns.              |
| **ReLU Activation**| -                                       | (48, 27, 37)       |                                                   |
| **Conv2D (4)**     | Filters: 64, Kernel: 3x3, Stride: 2     | (64, 13, 18)       | Refines feature representations.                 |
| **ReLU Activation**| -                                       | (64, 13, 18)       |                                                   |
| **Conv2D (5)**     | Filters: 64, Kernel: 3x3, Stride: 1     | (64, 11, 16)       | Enhances spatial resolution further.             |
| **ReLU Activation**| -                                       | (64, 11, 16)       |                                                   |
| **Flatten**        | -                                       | (11264)            | Flattens spatial features into a 1D vector.      |
| **Linear (1)**     | Input: 11264, Output: 512               | (512)              | Reduces dimensionality, learning high-level concepts. |
| **ReLU Activation**| -                                       | (512)              |                                                   |
| **Linear (2)**     | Input: 512, Output: 100                 | (100)              | Further compression of features.                 |
| **ReLU Activation**| -                                       | (100)              |                                                   |
| **Linear (3)**     | Input: 100, Output: 50                  | (50)               | Captures finer details in the representation.    |
| **ReLU Activation**| -                                       | (50)               |                                                   |
| **Output Layer**   | Input: 50, Output: 9                    | (9)                | Predicts the driving command vector.             |

---

## Prerequisites

### Software
- **CARLA Simulator**: Tested on version 0.9.10.1
- **Python**: 3.8.

# Modular Pipeline

An implentation of a modular pipeline for autonomous driving based on a front view image
![image](https://github.com/user-attachments/assets/6625c17f-2a63-441c-bc31-42b10e095c45)

## lane_detection.py

The front view image is converted to birds eye view, converted to grayscale, then has its absolute gradients calculated. The rowise maximas are found and used as the outline for the lanes. The first lane points are used for the lanes. Each maxima is placed into either the left or right lane. 
![image](https://github.com/user-attachments/assets/df6e2d39-4b92-45c3-8171-f98e1e6c7084)
![image](https://github.com/user-attachments/assets/312424ab-9cbc-4f24-9ed1-9f691656c0a8)
![image](https://github.com/user-attachments/assets/8cefd90e-6c73-4499-a6db-e1f97fd95e0b)

## waypoint.py

I used scipy to create splines for our lanes. We take N equidistant points for each spline. We take the midpoint between these points to find our waypoints. 
![image](https://github.com/user-attachments/assets/14ec9f95-b71c-4df4-8376-2dad79b0de8a)
![image](https://github.com/user-attachments/assets/ba45bd16-8d04-483e-8893-9476a5eb7cd0)

## lateral_control.py

I incorporated a stanely controller. A stanely controller is a nonlinear controller used for path following that is meant to minimize lateral air between the current postion and desired trajectory. 
![image](https://github.com/user-attachments/assets/8cd45c02-e623-4e66-bc8c-66e5fc4cbccb)
![image](https://github.com/user-attachments/assets/42364659-0f7c-426e-9a1f-9046903e188f)

## longitundinal_control.py

For longitundinal control, I incoporated a basic PID controller. 

![image](https://github.com/user-attachments/assets/f678d830-44c7-4f37-b43c-501ca22ab940)

# Deep Q-Learning Agent

## Overview

This project implements a Deep Q-Learning (DQL) agent for the CARLA autonomous driving simulator. The agent is trained to learn an optimal driving policy by interacting with the environment, leveraging techniques like experience replay and fixed Q-targets to improve stability and performance.

---

## Features

1. **Deep Q-Network (DQN)**  
   - A neural network processes a single input frame and predicts Q-values for all possible actions.  
   - The network architecture is designed to effectively extract spatial features, enabling the agent to make informed decisions.

2. **Deep Q-Learning Functions**  
   - `perform_qlearning_step`: Implements a single Q-Learning update step, including gradient clipping for training stability.  
   - `update_target_net`: Periodically synchronizes the target network with the policy network to ensure consistent learning.

3. **Action Selection**  
   - Supports ϵ-greedy exploration and greedy exploitation strategies for action selection.  
   - Balances exploration (to discover new strategies) and exploitation (to optimize learned strategies).

4. **Training and Evaluation**  
   - The agent is trained using a replay memory buffer to sample diverse experiences, improving learning stability.  
   - Loss and reward curves are generated to monitor training progress.  
   - The trained agent is evaluated to assess its performance in navigating the CARLA environment.

---

## Observations and Results

1. **Training Progress**  
   - The agent learns to consistently achieve positive rewards over time as ϵ-greedy exploration decreases.  
   - Loss curves indicate a gradual stabilization as training progresses, differing from the trends observed in standard supervised learning tasks.

2. **Evaluation**  
   - The trained agent performs well in most scenarios but struggles in specific situations requiring complex maneuvering.  
   - Observations provide insights into potential areas for improvement, such as enhancing the model architecture or refining hyperparameters.

---

## Conclusion

This project demonstrates the implementation and training of a Deep Q-Learning agent in the CARLA environment. By combining a well-designed DQN, effective action selection strategies, and robust training mechanisms, the agent successfully learns to navigate and make decisions in a simulated environment.
