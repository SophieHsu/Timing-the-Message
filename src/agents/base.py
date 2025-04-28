import torch.nn as nn

class BaseAgent(nn.Module):
    """Base class for all agents"""
    def __init__(self, args):
        super().__init__()
        self.args = args

    def get_value(self, x):
        """Get value estimate for given input"""
        raise NotImplementedError

    def get_action_and_value(self, x, action=None):
        """Get action, log probability, entropy, and value for given input"""
        raise NotImplementedError

    def forward(self, x):
        """Forward pass through the network"""
        raise NotImplementedError 