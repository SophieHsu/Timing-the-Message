import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNModule(nn.Module):
    def __init__(self, input_channels=26, input_height=7, input_width=8, output_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the flattened size after CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, input_height, input_width)
            sample_output = self.cnn(sample_input)
            self.flattened_size = sample_output.shape[1]
            
        self.fc = layer_init(nn.Linear(self.flattened_size, output_dim))
        
    def forward(self, x):
        # Reshape input if needed (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        # Ensure input is in the correct format (batch_size, channels, height, width)
        if len(x.shape) == 4 and x.shape[1] != 26:
            # If channels are not in the second dimension, transpose
            x = x.permute(0, 3, 1, 2)
            
        x = self.cnn(x)
        x = self.fc(x)
        return x


class SteakhouseFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the steakhouse environment that uses a CNN to process (7x8x26) input data.
    
    :param observation_space: (gym.Space) The observation space
    :param features_dim: (int) Number of features extracted
    """
    
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim=features_dim)
        self.cnn = CNNModule(output_dim=features_dim)
        
    def forward(self, observations):
        return self.cnn(observations)


class SteakhouseNotifierFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the steakhouse environment with notification capabilities.
    
    :param args: Arguments containing environment configuration
    :param observation_space: (gym.Space) The observation space
    :param features_dim: (int) Number of features extracted
    """
    
    def __init__(self, args, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim=features_dim)
        self.args = args
        self.observation_space = np.array(observation_space).prod()
        self.cnn = CNNModule(output_dim=features_dim)
        
    def forward(self, observations):
        # Reshape observations to separate the image data from notification data
        # Assuming the first part is the image data (7x8x26) and the rest is notification data
        observations = observations.reshape(-1, self.args.human_utterance_memory_length, self.observation_space+self.args.noti_action_length)
        image_data = observations[:,:,:-self.args.noti_action_length]
        noti_data = observations[:, :, -self.args.noti_action_length:]
        
        # Reshape image data for CNN processing
        # Reshape to (batch_size * human_utterance_memory_length, 7, 8, 26)
        image_data = image_data.reshape(-1, 7, 8, 26)
        
        # Process through CNN
        features = self.cnn(image_data)
        features = features.reshape(-1, self.args.human_utterance_memory_length, self.args.steakhouse_feature_dim)

        # Concatenate with notification data
        features = torch.cat([features, noti_data], dim=-1).reshape(-1, self.args.human_utterance_memory_length*(self.args.steakhouse_feature_dim+self.args.noti_action_length))
        
        return features 