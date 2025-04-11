import torch
import numpy as np
from agents.base import BaseAgent

class HeuristicAgent(BaseAgent):
    """Heuristic-based notifier agent that computes time-to-collision and outputs notifications"""
    
    def __init__(self, envs, args):
        super().__init__(envs, args)
        self.danger_threshold = 0.3  # Distance threshold to consider danger zones
        self.ttc_threshold = 10  # Time-to-collision threshold in steps
        self.notification_length = 2  # Default notification length
        self.prev_notification = None
        
    def get_value(self, x):
        """Get value estimate for given input - not used for heuristic agent"""
        return torch.zeros(x.shape[0], 1)
        
    def get_action_and_value(self, x, action=None):
        """Get notification action and value for given input"""
        # Extract danger zone distances from state
        danger_distances = x[:, 8:12]  # [left, right, top, bottom] danger zone distances
        
        # Compute time-to-collision for each danger zone
        velocities = x[:, 2:4]  # [x_vel, y_vel]
        ttc = torch.zeros_like(danger_distances)
        
        # Only compute TTC for zones we're moving towards
        for i in range(4):
            if i < 2:  # Left/right zones
                if velocities[:, 0] != 0:  # Moving horizontally
                    ttc[:, i] = danger_distances[:, i] / velocities[:, 0]
            else:  # Top/bottom zones
                if velocities[:, 1] != 0:  # Moving vertically
                    ttc[:, i] = danger_distances[:, i] / velocities[:, 1]
        
        # Find the most critical danger zone (lowest TTC)
        min_ttc, min_ttc_idx = torch.min(ttc, dim=1)
        
        # Initialize notification action
        batch_size = x.shape[0]
        notification = torch.zeros((batch_size, 3), dtype=torch.long)
        
        # For each environment in the batch
        for i in range(batch_size):
            # Check if we're in immediate danger
            in_danger = torch.any(danger_distances[i] < 0)
            
            # Check if we're approaching danger
            approaching_danger = min_ttc[i] < self.ttc_threshold and min_ttc[i] > 0
            
            if in_danger or approaching_danger:
                # Determine which action to take based on the danger zone
                action_type = 2  # New notification
                action = min_ttc_idx[i].item()
                
                # Set notification length based on urgency
                if in_danger:
                    length = 5  # Longer notification for immediate danger
                else:
                    length = 2  # Shorter notification for approaching danger
                    
                notification[i] = torch.tensor([action_type, action, length])
            else:
                # No danger, either no notification or continue previous
                if self.prev_notification is not None:
                    notification[i] = torch.tensor([1, 0, 0])  # Continue previous
                else:
                    notification[i] = torch.tensor([0, 0, 0])  # No notification
                    
        self.prev_notification = notification.clone()
        
        # Return dummy values for compatibility with base class
        return notification, torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size, 1)
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.get_action_and_value(x)[0]
