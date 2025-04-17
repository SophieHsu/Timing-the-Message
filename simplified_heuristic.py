import torch
import numpy as np
import math
import matplotlib.pyplot as plt

class SimplifiedHeuristicAgent:
    def __init__(self, num_envs=2, device='cpu'):
        self.num_envs = num_envs
        self.device = device
    
    def get_action_and_value(self, x, optimal_path=None):
        """
        Simplified version of the get_action_and_value function.
        
        Args:
            x: Tensor of shape [batch_size, memory_length, feature_dim]
            optimal_path: List of (x, y) coordinates representing the optimal path
            
        Returns:
            action: Selected action for each environment
            logprob: Log probability of the action
            entropy: Entropy of the action distribution
            value: Value estimate
        """
        # Reshape the input
        x = x.reshape(self.num_envs, -1, x.shape[-1])
        latest_obs = x[:, -1, :].reshape(self.num_envs, -1)
        
        # Extract past trajectory and danger zone distances
        past_trajectory = x[:, :, :2]
        to_left_danger_zone_distance = latest_obs[:, 8]
        to_right_danger_zone_distance = latest_obs[:, 9]
        to_top_danger_zone_distance = latest_obs[:, 10]
        to_bottom_danger_zone_distance = latest_obs[:, 11]
        
        # Initialize action_id
        action_id = torch.zeros(self.num_envs).to(self.device)
        
        # Check for danger zones
        action_id = torch.where((action_id == 0) & (to_top_danger_zone_distance < 0.3), 
                              torch.tensor(0).to(self.device), action_id)
        action_id = torch.where((action_id == 0) & (to_right_danger_zone_distance < 0.3), 
                              torch.tensor(1).to(self.device), action_id)
        action_id = torch.where((action_id == 0) & (to_bottom_danger_zone_distance < 0.3), 
                              torch.tensor(2).to(self.device), action_id)
        action_id = torch.where((action_id == 0) & (to_left_danger_zone_distance < 0.3), 
                              torch.tensor(3).to(self.device), action_id)
        
        # If optimal path is provided and not in danger, follow the optimal path
        if optimal_path is not None:
            # Get current position from the latest observation
            current_pos = (latest_obs[:, 0], latest_obs[:, 1])
            
            # Find the nearest point in the optimal path to the current position
            min_distances = []
            nearest_indices = []
            
            for i in range(self.num_envs):
                min_dist = float('inf')
                nearest_idx = 0
                
                for j, path_point in enumerate(optimal_path):
                    dist = math.sqrt((current_pos[0][i] - path_point[0])**2 + 
                                    (current_pos[1][i] - path_point[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = j
                
                min_distances.append(min_dist)
                nearest_indices.append(nearest_idx)
            
            # For each environment, determine the action based on the optimal path
            for i in range(self.num_envs):
                # Skip if we're already in danger mode
                if action_id[i] in [1, 2, 3]:
                    continue
                
                # Get the nearest point and a few points ahead in the path
                nearest_idx = nearest_indices[i]
                look_ahead = min(5, len(optimal_path) - nearest_idx - 1)
                
                if look_ahead > 0:
                    # Get the target point (a few steps ahead in the path)
                    target_idx = nearest_idx + look_ahead
                    target_point = optimal_path[target_idx]
                    
                    # Calculate the direction to the target
                    dx = target_point[0] - current_pos[0][i]
                    dy = target_point[1] - current_pos[1][i]
                    
                    # Calculate how deviated the past trajectory is from the optimal path
                    # Get the last few positions from the past trajectory
                    trajectory_length = min(10, past_trajectory.shape[1])
                    if trajectory_length > 1:
                        # Calculate the average deviation from the optimal path
                        total_deviation = 0
                        for t in range(trajectory_length):
                            # Find the nearest point in the optimal path for each position in the trajectory
                            min_traj_dist = float('inf')
                            for path_point in optimal_path:
                                dist = math.sqrt((past_trajectory[i, t, 0] - path_point[0])**2 + 
                                                (past_trajectory[i, t, 1] - path_point[1])**2)
                                if dist < min_traj_dist:
                                    min_traj_dist = dist
                            total_deviation += min_traj_dist
                        
                        avg_deviation = total_deviation / trajectory_length
                        
                        # If the deviation is too high, prioritize getting back on track
                        if avg_deviation > 0.5:  # Threshold for significant deviation
                            # Determine the action based on the direction to the nearest point
                            if abs(dx) > abs(dy):
                                # Horizontal movement is more significant
                                if dx > 0:
                                    action_id[i] = 1  # right
                                else:
                                    action_id[i] = 3  # left
                            else:
                                # Vertical movement is more significant
                                if dy > 0:
                                    action_id[i] = 0  # up
                                else:
                                    action_id[i] = 2  # down
                        else:
                            # If we're reasonably on track, use a more nuanced approach
                            # Calculate the heading direction of the optimal path
                            if nearest_idx + 1 < len(optimal_path):
                                path_dx = optimal_path[nearest_idx + 1][0] - optimal_path[nearest_idx][0]
                                path_dy = optimal_path[nearest_idx + 1][1] - optimal_path[nearest_idx][1]
                                
                                # Calculate the heading direction of the past trajectory
                                if trajectory_length > 1:
                                    traj_dx = past_trajectory[i, -1, 0] - past_trajectory[i, -2, 0]
                                    traj_dy = past_trajectory[i, -1, 1] - past_trajectory[i, -2, 1]
                                    
                                    # Calculate the angle difference between the two headings
                                    path_angle = math.atan2(path_dy, path_dx)
                                    traj_angle = math.atan2(traj_dy, traj_dx)
                                    angle_diff = abs(path_angle - traj_angle)
                                    
                                    # If the angle difference is significant, adjust the heading
                                    if angle_diff > 0.5:  # About 30 degrees
                                        # Determine which action would help align with the optimal path
                                        if path_angle > traj_angle:
                                            # Need to turn counterclockwise
                                            if path_angle - traj_angle < math.pi:
                                                action_id[i] = 3  # left
                                            else:
                                                action_id[i] = 1  # right
                                        else:
                                            # Need to turn clockwise
                                            if traj_angle - path_angle < math.pi:
                                                action_id[i] = 1  # right
                                            else:
                                                action_id[i] = 3  # left
                                    else:
                                        # We're heading in roughly the right direction, continue
                                        if abs(dx) > abs(dy):
                                            # Horizontal movement is more significant
                                            if dx > 0:
                                                action_id[i] = 1  # right
                                            else:
                                                action_id[i] = 3  # left
                                        else:
                                            # Vertical movement is more significant
                                            if dy > 0:
                                                action_id[i] = 0  # up
                                            else:
                                                action_id[i] = 2  # down
                            else:
                                # If we're at the end of the path, just move toward the target
                                if abs(dx) > abs(dy):
                                    # Horizontal movement is more significant
                                    if dx > 0:
                                        action_id[i] = 1  # right
                                    else:
                                        action_id[i] = 3  # left
                                else:
                                    # Vertical movement is more significant
                                    if dy > 0:
                                        action_id[i] = 0  # up
                                    else:
                                        action_id[i] = 2  # down
                    else:
                        # If we don't have enough trajectory history, just move toward the target
                        if abs(dx) > abs(dy):
                            # Horizontal movement is more significant
                            if dx > 0:
                                action_id[i] = 1  # right
                            else:
                                action_id[i] = 3  # left
                        else:
                            # Vertical movement is more significant
                            if dy > 0:
                                action_id[i] = 0  # up
                            else:
                                action_id[i] = 2  # down
        
        # Convert action_id to tensor
        action = action_id.clone().detach().long().to(self.device)
        
        # For simplicity, return dummy values for logprob, entropy, and value
        logprob = torch.zeros(self.num_envs).to(self.device)
        entropy = torch.zeros(self.num_envs).to(self.device)
        value = torch.zeros(self.num_envs).to(self.device)
        
        return action, logprob, entropy, value

def create_sample_data(num_envs=2, memory_length=5):
    """Create sample observation data for testing."""
    # Create a tensor with shape [batch_size, memory_length, feature_dim]
    # feature_dim includes: x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact, 
    # to_left_danger, to_right_danger, to_top_danger, to_bottom_danger
    batch_size = num_envs
    feature_dim = 12
    
    # Create random data
    data = np.random.randn(batch_size, memory_length, feature_dim)
    
    # Set some specific values for testing
    # Set danger zone distances
    data[:, :, 8] = 0.8  # to_left_danger
    data[:, :, 9] = 0.7  # to_right_danger
    data[:, :, 10] = 0.6  # to_top_danger
    data[:, :, 11] = 0.5  # to_bottom_danger
    
    # Set the latest observation to have one danger zone close
    data[0, -1, 10] = 0.2  # First environment is close to top danger zone
    data[1, -1, 9] = 0.25  # Second environment is close to right danger zone
    
    # Convert to tensor
    return torch.tensor(data, dtype=torch.float32)

def create_sample_optimal_path():
    """Create a sample optimal path for testing."""
    # Create a simple path from top to bottom with some curves
    path = []
    
    # Start at the top center
    path.append((0.0, 1.0))
    
    # Add some points to create a path
    path.append((0.2, 0.8))
    path.append((0.4, 0.6))
    path.append((0.2, 0.4))
    path.append((0.0, 0.2))
    path.append((0.0, 0.0))  # Goal at the bottom center
    
    return path

def create_sample_past_trajectory(num_envs=2, memory_length=5):
    """Create a sample past trajectory that deviates from the optimal path."""
    # Create a tensor with shape [batch_size, memory_length, 2] (x, y coordinates)
    batch_size = num_envs
    
    # Create data that deviates from the optimal path
    data = np.zeros((batch_size, memory_length, 2))
    
    # First environment: trajectory that follows the optimal path
    data[0, 0] = [0.0, 1.0]
    data[0, 1] = [0.1, 0.9]
    data[0, 2] = [0.2, 0.8]
    data[0, 3] = [0.3, 0.7]
    data[0, 4] = [0.4, 0.6]
    
    # Second environment: trajectory that deviates from the optimal path
    data[1, 0] = [0.0, 1.0]
    data[1, 1] = [0.1, 0.9]
    data[1, 2] = [0.2, 0.8]
    data[1, 3] = [0.3, 0.9]  # Deviates upward
    data[1, 4] = [0.4, 1.0]  # Continues to deviate
    
    return torch.tensor(data, dtype=torch.float32)

def plot_test_results(optimal_path, past_trajectory, action_id):
    """Plot the optimal path, past trajectory, and selected actions."""
    plt.figure(figsize=(10, 8))
    
    # Plot optimal path
    path_x = [p[0] for p in optimal_path]
    path_y = [p[1] for p in optimal_path]
    plt.plot(path_x, path_y, 'b-', label='Optimal Path')
    plt.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
    plt.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='Goal')
    
    # Plot past trajectories
    colors = ['g', 'r']
    for i in range(past_trajectory.shape[0]):
        traj_x = past_trajectory[i, :, 0].numpy()
        traj_y = past_trajectory[i, :, 1].numpy()
        plt.plot(traj_x, traj_y, f'{colors[i]}-', label=f'Trajectory {i+1}')
        plt.plot(traj_x[-1], traj_y[-1], f'{colors[i]}o', markersize=8)
    
    # Add action annotations
    action_names = ['Up', 'Right', 'Down', 'Left']
    for i in range(len(action_id)):
        action_idx = int(action_id[i])  # Convert to integer
        if 0 <= action_idx <= 3:  # Check if action is valid
            plt.annotate(f'Action: {action_names[action_idx]}', 
                        xy=(past_trajectory[i, -1, 0], past_trajectory[i, -1, 1]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.title('Test Results: Optimal Path, Past Trajectory, and Selected Actions')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.savefig('test_results.png')
    plt.show()

def test_simplified_heuristic():
    """Test the simplified get_action_and_value function with sample data."""
    # Create a SimplifiedHeuristicAgent instance
    agent = SimplifiedHeuristicAgent(num_envs=2)
    
    # Create sample data
    x = create_sample_data()
    optimal_path = create_sample_optimal_path()
    past_trajectory = create_sample_past_trajectory()
    
    # Replace the past trajectory in the observation data
    x[:, :, :2] = past_trajectory
    
    # Call the function
    action, logprob, entropy, value = agent.get_action_and_value(x, optimal_path)
    
    # Print the results
    print("Selected actions:", action)
    print("Log probabilities:", logprob)
    print("Entropy:", entropy)
    print("Value:", value)
    
    # Plot the results
    plot_test_results(optimal_path, past_trajectory, action.numpy())
    
    return action, logprob, entropy, value

if __name__ == "__main__":
    test_simplified_heuristic() 