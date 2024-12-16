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
        if image is None:
            image = torch.zeros(3, 240, 320)
            control = {'steering': 0, 'throttle': 0, 'brake': 0}
        else:
            image = self.transform(image)

        # One hot envoded action
        # throttle, throttle left, throttle right, brake, coast
        action = torch.zeros(5)

        steer = control['steering'] # right is positive, left is negative
        throttle = control['throttle']
        brake = control['brake']

        if steer < 0:
            action[0] = 1
        elif steer > 0:
            action[1] = 1
        elif throttle > 0:
            action[2] = 1
        elif brake > 0:
            action[3] = 1
        else:
            action[4] = 1

        # assert torch.sum(action) == 1
        assert image.shape == (3, 240, 320)

        return image, action

def get_dataloader(data_dir, batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    