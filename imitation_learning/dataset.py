import glob

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.npy') #need to change to your data format

        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        data = np.load(self.data_list[idx], allow_pickle=True).item()

        observation = data['observation']
        control = data['action']

        image = observation['camera']
        if self.transform:
            image = self.transform(image)

        # One hot envoded action
        # control: [steering: float, throttle: float, brake: float, hand_brake: bool, reverse: bool]
        # Actions: 
        # [throttle, steer_left and throttle, steer_right and throttle, 
        # brake,  brake and steer_left, brake and steer_right, 
        # coast, steer_left, steer_right, 
        action = torch.zeros(9)

        steering = control['steering'] # right is positive, left is negative
        throttle = control['throttle']
        brake = control['brake']

        if throttle > 0 and brake == 0: # throttle
            if steering < 0:
                action[1] = 1 # steer_left and throttle
            elif steering > 0:
                action[2] = 1 # steer_right and throttle
            else:
                action[0] = 1 # throttle
        elif throttle == 0 and brake > 0: # brake
            if steering < 0:
                action[4] = 1 # brake and steer_left
            elif steering > 0:
                action[5] = 1 # brake and steer_right
            else:
                action[3] = 1 # brake
        elif throttle == 0 and brake == 0: # coast
            if steering < 0:
                action[7] = 1 # steer_left
            elif steering > 0:
                action[8] = 1 # steer_right 
            else:
                action[6] = 1 # coast

        # Double check that the action is valid
        assert torch.sum(action) == 1
        assert image.shape == (3, 240, 320)

        return image, action

def get_dataloader(data_dir, batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    