import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size


        # Old model
        # selfv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=0),
        #     nn.BatchNorm2d(32),  # Add batch normalization
        #     nn.ReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        
        # # fully connected layers
        # self.fc1 = nn.Sequential(
        #     nn.Linear(59904, 2048),
        #     nn.ReLU()
        # )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(2048, 512),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Linear(512, action_size) .con

        # New model
        self.model = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11264 , 512),
            nn.ReLU(),
            nn.Linear(512 , 100),
            nn.ReLU(),
            nn.Linear(100 , 50),
            nn.ReLU(),
            nn.Linear(50, action_size)
        ).to(device)

    def forward(self, x):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            image observation (according to Q3.1)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        # TODO: Forward pass through the network

        return self.model(x)


