import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium_envs.envs.lunar_lander import DangerZoneLunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN
import heapq
import math
import torch
from torch.distributions.categorical import Categorical

class HeuristicAgent:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.danger_zones = env.envs[0].unwrapped.danger_zones
        self.W = VIEWPORT_W / SCALE
        self.H = VIEWPORT_H / SCALE
        self.helipad_x1 = None
        self.helipad_x2 = None
        self.helipad_y = None
        self.goal = None
        self.landing_pad = None
        self.optimal_path = None  # Initialize optimal_path attribute
        self.previous_utterances = None

        # Create the environment map
        self.create_env_map(env)
        
        # Plot the environment map
        self.plot_env_map()
        
        # Compute the optimal path from the starting position
        start_pos = (VIEWPORT_W / SCALE / 2, VIEWPORT_H / SCALE - 1)  # Top center
        self.optimal_path = self.compute_optimal_path(start_pos)
        
        # Plot the optimal path
        self.plot_optimal_path()

    def create_env_map(self, env):
        """
        Create a map representation of the environment including danger zones and landing pad.
        
        Args:
            env: The DangerZoneLunarLander environment
        """
        self.env = env
        self.W = VIEWPORT_W / SCALE
        self.H = VIEWPORT_H / SCALE
        
        # Create terrain parameters
        CHUNKS = 11
        chunk_x = [self.W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = self.H / 4
        
        # Set landing pad coordinates (x, y, width, height)
        self.landing_pad = (
            self.helipad_x1,  # x position
            self.helipad_y,   # y position
            self.helipad_x2 - self.helipad_x1,  # width
            0.1  # height (small value for visualization)
        )
        
        return None  # No need to return a grid map anymore

    def plot_env_map(self):
        """
        Plot the environment map with danger zones and landing pad.
        """
        plt.figure(figsize=(10, 10))
        
        # Get dimensions
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        
        # Create terrain
        CHUNKS = 11
        height = np.full(CHUNKS, H / 4)  # Constant height for flat terrain
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        helipad_y = H / 4
        
        # Plot terrain
        plt.plot(chunk_x, height, 'k-', linewidth=2)
        
        # Plot landing pad (green)
        pad_x = [helipad_x1, helipad_x2]
        pad_y = [helipad_y, helipad_y]
        plt.plot(pad_x, pad_y, 'g-', linewidth=4)
        
        # Plot flag poles
        for x in [helipad_x1, helipad_x2]:
            flagy1 = helipad_y
            flagy2 = flagy1 + 2
            plt.plot([x, x], [flagy1, flagy2], 'k-', linewidth=1)  # Changed from white to black for visibility
            plt.fill([x, x, x + 0.5], [flagy2, flagy2 - 0.2, flagy2 - 0.1], 'y')
        
        # Add danger zones
        for zone in self.danger_zones:
            x_min = zone[0][0]
            x_max = zone[0][1]
            y_min = zone[1][0]
            y_max = zone[1][1]
            
            # Convert normalized coordinates to world coordinates
            x_min = (x_min + 1) * W/2
            x_max = (x_max + 1) * W/2
            y_min = y_min * H/2 + helipad_y
            y_max = y_max * H/2 + helipad_y
            
            width = x_max - x_min
            height = y_max - y_min
            
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), width, height, 
                                             fill=True, color='red', alpha=0.3))
        
        # Set plot limits and labels
        plt.xlim(0, W)
        plt.ylim(0, H)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Environment Map with Danger Zones and Landing Pad')
        plt.grid(True)
        
        # Set background color to black to match game
        plt.gca().set_facecolor('black')
        plt.gcf().set_facecolor('black')
        
        # Save the plot
        plt.savefig('env_map.png', facecolor='black', edgecolor='black')
        plt.show()

    def compute_optimal_path(self, start):
        """
        Compute the optimal path from start to goal using A* algorithm.
        
        Args:
            start: The starting position (x, y)
            
        Returns:
            The optimal path as a list of (x, y) coordinates
        """
        # Convert start position to world coordinates
        start_x = start[0]
        start_y = start[1]
        start_pos = (start_x, start_y)

        # Goal is the center of the landing pad
        goal_pos = ((self.helipad_x1 + self.helipad_x2) / 2, self.helipad_y)
        
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        
        # Initialize g_score and f_score dictionaries
        g_score = {start_pos: 0}
        f_score = {start_pos: self.get_heuristic_for_Astar(start_pos, goal_pos)}
        
        # Initialize came_from dictionary
        came_from = {}
        
        # Add start position to open set
        heapq.heappush(open_set, (f_score[start_pos], start_pos))
        
        # A* algorithm
        while open_set:
            # Get the position with the lowest f_score
            current_f, current_pos = heapq.heappop(open_set)
            
            # If we reached the goal, reconstruct the path
            if self.is_goal_reached(current_pos, goal_pos):
                path = []
                while current_pos in came_from:
                    x = (current_pos[0] - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
                    y = (current_pos[1] - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)
                    path.append((x, y))
                    current_pos = came_from[current_pos]

                x = (start_pos[0] - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
                y = (start_pos[1] - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)
                path.append((x, y))
                path.reverse()
                self.optimal_path = path
                return path
            
            # Add current position to closed set
            closed_set.add(current_pos)
            
            # Check all neighbors
            for dx, dy in [(0, 0.1), (0.1, 0), (0, -0.1), (-0.1, 0), 
                          (0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1)]:
                neighbor_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Skip if neighbor is out of bounds
                if not self.is_valid_position(neighbor_pos):
                    continue
                
                # Skip if neighbor is in a danger zone
                if self.is_in_danger_zone(neighbor_pos):
                    continue
                
                # Skip if neighbor is in closed set
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current_pos] + math.sqrt(dx**2 + dy**2)
                
                # If neighbor is not in open set or tentative g_score is better, update
                if neighbor_pos not in g_score or tentative_g_score < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current_pos
                    g_score[neighbor_pos] = tentative_g_score
                    f_score[neighbor_pos] = g_score[neighbor_pos] + self.get_heuristic_for_Astar(neighbor_pos, goal_pos)
                    
                    # Add neighbor to open set if not already there
                    if neighbor_pos not in [pos for _, pos in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor_pos], neighbor_pos))
        
        # If we get here, no path was found
        print("No path found!")
        return None

    def is_valid_position(self, pos):
        """Check if position is within bounds"""
        x, y = pos
        return 0 <= x <= self.W and 0 <= y <= self.H

    def is_in_danger_zone(self, pos):
        """Check if position is in any danger zone or its buffer area"""
        x, y = pos
        # Convert to normalized coordinates
        x_norm = (x / (self.W/2)) - 1
        y_norm = (y - self.helipad_y) / (self.H/2)
        
        # Buffer size in normalized coordinates
        buffer_size = 0.1  # Adjust this value to change the buffer size
        
        for zone in self.danger_zones:
            x_min = zone[0][0] - buffer_size
            x_max = zone[0][1] + buffer_size
            y_min = zone[1][0] - buffer_size
            y_max = zone[1][1] + buffer_size
            
            if (x_min <= x_norm <= x_max and y_min <= y_norm <= y_max):
                return True
        return False

    def is_goal_reached(self, current_pos, goal_pos):
        """Check if current position is close enough to goal"""
        dx = current_pos[0] - goal_pos[0]
        dy = current_pos[1] - goal_pos[1]
        return math.sqrt(dx*dx + dy*dy) < 0.1  # Within 0.1 units of goal

    def get_heuristic_for_Astar(self, current_pos, goal_pos):
        """
        Calculate the heuristic for A* algorithm.
        
        Args:
            current_pos: The current position in world coordinates
            goal_pos: The goal position in world coordinates
            
        Returns:
            The heuristic value
        """
        # Calculate distance to goal (Euclidean distance)
        dx = current_pos[0] - goal_pos[0]
        dy = current_pos[1] - goal_pos[1]
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        # Calculate distance to danger zones
        min_distance_to_danger = float('inf')
        x_norm = (current_pos[0] / (self.W/2)) - 1
        y_norm = (current_pos[1] - self.helipad_y) / (self.H/2)
        
        for zone in self.danger_zones:
            x_min = zone[0][0]
            x_max = zone[0][1]
            y_min = zone[1][0]
            y_max = zone[1][1]
            
            # Calculate distance to danger zone
            if x_norm < x_min:
                dx = x_min - x_norm
            elif x_norm > x_max:
                dx = x_norm - x_max
            else:
                dx = 0
                
            if y_norm < y_min:
                dy = y_min - y_norm
            elif y_norm > y_max:
                dy = y_norm - y_max
            else:
                dy = 0
                
            distance = math.sqrt(dx**2 + dy**2)
            min_distance_to_danger = min(min_distance_to_danger, distance)
        
        # Combine distance to goal and distance to danger zones
        # We want to minimize distance to goal while maximizing distance to danger zones
        heuristic = distance_to_goal - 0.5 * min_distance_to_danger
        
        return heuristic

    def plot_optimal_path(self):
        if not self.optimal_path:
            print("Optimal path not computed yet. Call compute_optimal_path first.")
            return
        
        plt.figure(figsize=(10, 10))
        
        # Create terrain
        CHUNKS = 11
        height = np.full(CHUNKS, self.H / 4)  # Constant height for flat terrain
        chunk_x = [self.W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        
        # Plot terrain
        plt.plot(chunk_x, height, 'k-', linewidth=2)
        
        # Plot landing pad (green)
        pad_x = [self.helipad_x1, self.helipad_x2]
        pad_y = [self.helipad_y, self.helipad_y]
        plt.plot(pad_x, pad_y, 'g-', linewidth=4)
        
        # Plot flag poles
        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 2
            plt.plot([x, x], [flagy1, flagy2], 'k-', linewidth=1)
            plt.fill([x, x, x + 0.5], [flagy2, flagy2 - 0.2, flagy2 - 0.1], 'y')
        
        # Add danger zones
        for zone in self.danger_zones:
            x_min = zone[0][0]
            x_max = zone[0][1]
            y_min = zone[1][0]
            y_max = zone[1][1]
            
            # Convert normalized coordinates to world coordinates
            x_min = (x_min + 1) * self.W/2
            x_max = (x_max + 1) * self.W/2
            y_min = y_min * self.H/2 + self.helipad_y
            y_max = y_max * self.H/2 + self.helipad_y
            
            width = x_max - x_min
            height = y_max - y_min
            
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), width, height, 
                                             fill=True, color='red', alpha=0.3))
        
        # Plot the optimal path
        path_x = [p[0] * (VIEWPORT_W / SCALE / 2) + VIEWPORT_W / SCALE / 2 for p in self.optimal_path]
        path_y = [p[1] * (VIEWPORT_H / SCALE / 2) + (self.helipad_y + LEG_DOWN / SCALE) for p in self.optimal_path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Optimal Path')
        
        # Plot start and goal positions
        plt.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
        plt.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='Goal')
        
        # Set plot limits and labels
        plt.xlim(0, self.W)
        plt.ylim(0, self.H)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Optimal Path for Lunar Lander')
        plt.grid(True)
        plt.legend()

        plt.savefig('optimal_path.png')
        plt.show()

    def get_action_and_value(self, x):
        latest_obs = x.reshape(self.args.num_envs, -1)

        # Based on the latest observation, retreive the closest distance to the danger zone
        current_pos = x[:, :2]
        to_left_danger_zone_distance = latest_obs[:, 8].cpu().numpy()
        to_right_danger_zone_distance = latest_obs[:, 9].cpu().numpy()
        to_top_danger_zone_distance = latest_obs[:, 10].cpu().numpy()
        to_bottom_danger_zone_distance = latest_obs[:, 11].cpu().numpy()

        # if any distance is less than 0.3, then action is based on the distance to the danger zone
        action_id = torch.tensor(np.array([-1]*self.args.num_envs), dtype=torch.long).to(self.args.device)
        close_to_danger_zone = np.where((to_right_danger_zone_distance < 0.5) | (to_bottom_danger_zone_distance < 0.5) | (to_left_danger_zone_distance < 0.5))[0]

        nearest_point_indices = self.get_nearest_point_in_optimal_path(current_pos, self.optimal_path)
        
        # # For each environment, get the past optimal path segment
        # past_optimal_paths = []
        # for env_idx in range(self.args.num_envs):
        #     nearest_idx = nearest_point_indices[env_idx]
        #     start_idx = max(0, nearest_idx - 10)  # Ensure we don't go below 0
        #     past_optimal_paths.append(optimal_path[start_idx:nearest_idx])
        
        # # Calculate divergence for each environment
        # divergences = []
        # for env_idx in range(self.args.num_envs):
        #     # Extract the trajectory for this environment
        #     env_trajectory = past_trajectory[env_idx:env_idx+1]
        #     divergence = self.measure_past_trjectory_divergence(env_trajectory, past_optimal_paths[env_idx])
        #     divergences.append(divergence)
        
        # # Convert to numpy array for easier handling
        # divergences = np.array(divergences)
        
        # Check which environments have high divergence
        # high_divergence_envs = divergences > 0.1
        
        # For environments with high divergence, compute next heading direction
        if len(close_to_danger_zone) > 0:
            nearest_idx = nearest_point_indices[0]
            # Ensure we don't go beyond the end of the path
            next_idx = max(0, nearest_idx-3)
            next_optimal_point = self.optimal_path[next_idx]
            
            # Extract the trajectory for this environment
            heading_action = self.compute_next_heading_direction(current_pos, next_optimal_point)
            
            # Update action_id for this environment
            action_id[0] = heading_action
        
        # Get the action that has the maximum distance to the danger zone
        condition = torch.ones(self.args.num_envs, dtype=torch.long).to(self.args.device)
        length = torch.zeros(self.args.num_envs, dtype=torch.long).to(self.args.device)
        condition[close_to_danger_zone] = 2
        utter_flag = np.where((self.previous_utterances == 2))[0]
        condition[utter_flag] = 1
        action_id[utter_flag] = 0
        action = torch.stack([condition, action_id, length], dim=1)

        self.previous_utterances = condition.cpu().numpy()

        return action, None, None, None

    def get_nearest_point_in_optimal_path(self, current_pos, optimal_path):
        """
        Find the index of the point in the optimal path that is closest to the current position.
        
        Args:
            current_pos: Tensor of shape [num_envs, 2] containing current positions
            optimal_path: List of (x, y) coordinates representing the optimal path
            
        Returns:
            Index of the nearest point in the optimal path
        """
        if optimal_path is None or len(optimal_path) == 0:
            return 0
            
        # Initialize array to store nearest indices for each environment
        nearest_indices = np.zeros(self.args.num_envs, dtype=int)
        
        # For each environment, find the nearest point in the optimal path
        for env_idx in range(self.args.num_envs):
            min_dist = float('inf')
            nearest_idx = 0
            
            for i, point in enumerate(optimal_path):
                # Calculate Euclidean distance between current position and path point
                # dist = np.sqrt(np.sum((current_pos[env_idx].cpu().numpy() - np.array(point)) ** 2))
                dist = abs(current_pos[env_idx].cpu().numpy()[0] - point[0])
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            nearest_indices[env_idx] = nearest_idx
                
        return nearest_indices
        
    def measure_past_trjectory_divergence(self, past_trajectory, past_optimal_path):
        """
        Measure how much the past trajectory has diverged from the optimal path.
        
        Args:
            past_trajectory: Tensor of shape [1, memory_length, 2] containing past positions for a single environment
            past_optimal_path: List of (x, y) coordinates representing the optimal path segment
            
        Returns:
            Divergence value (higher means more divergence)
        """
        if past_optimal_path is None or len(past_optimal_path) == 0:
            return 0.0
            
        # Convert tensors to numpy arrays
        trajectory = past_trajectory.cpu().numpy()
        
        # Calculate the average distance between trajectory points and optimal path points
        total_dist = 0.0
        count = 0
        
        # For each point in the trajectory, find the closest point in the optimal path
        for i in range(trajectory.shape[1]):
            point = trajectory[0, i, :]  # Shape: [2]
            
            # Find the closest point in the optimal path
            min_dist = float('inf')
            for opt_point in past_optimal_path:
                dist = np.sqrt(np.sum((point - opt_point) ** 2))
                min_dist = min(min_dist, dist)
                
            total_dist += min_dist
            count += 1
            
        # Return the average distance as the divergence measure
        return total_dist / max(1, count)
        
    def compute_next_heading_direction(self, current_pos, next_optimal_point):
        """
        Compute the heading direction to reach the next optimal point.
        
        Args:
            past_trajectory: Tensor of shape [1, memory_length, 2] containing past positions for a single environment
            next_optimal_point: (x, y) coordinates of the next point to reach
            
        Returns:
            Action ID (0: up, 1: right, 2: down, 3: left)
        """
        # Calculate the vector from current position to the next optimal point
        direction_vector = np.array(next_optimal_point) - current_pos.cpu().numpy()[0]
        
        # Determine the primary direction based on the vector components
        # The action space is: 0: down, 1: right, 2: up, 3: left
        if abs(direction_vector[1]) > abs(direction_vector[0]):
            # Vertical movement is dominant
            if direction_vector[1] > 0:
                return 1  # Up
            else:
                return -1  # Down
        else:
            # Horizontal movement is dominant
            if direction_vector[0] > 0:
                return 2  # Right
            else:
                return 0  # Left

if __name__ == "__main__":
    # Create the environment
    env = gym.make("DangerZoneLunarLander", render_mode="rgb_array")
    
    # Initialize the heuristic agent
    heuristic = HeuristicAgent(env, args)
    
    # Create the environment map
    heuristic.create_env_map(env)
    
    # Plot the environment map
    heuristic.plot_env_map()
    
    # Compute the optimal path from the starting position
    start_pos = (VIEWPORT_W / SCALE / 2, VIEWPORT_H / SCALE - 1)  # Top center
    optimal_path = heuristic.compute_optimal_path(start_pos)
    
    # Plot the optimal path
    heuristic.plot_optimal_path()

    # Close the environment
    env.close()


