import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium_envs.envs.lunar_lander import DangerZoneLunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

def create_danger_zone_reward_heatmap(env, resolution=50, angle=0.0, velocity=(0, 0), angular_velocity=0.0, legs_contact=(0, 0)):
    """
    Create heatmaps of danger zone reward components for different positions in the environment.
    
    Args:
        env: The Lunar Lander environment
        resolution: Number of points in each dimension of the grid
        angle: Fixed angle for the lander
        velocity: Fixed velocity (vx, vy) for the lander
        angular_velocity: Fixed angular velocity for the lander
        legs_contact: Fixed legs contact state (left, right)
    
    Returns:
        reward_components: Dictionary of 2D arrays of reward components
        x_grid: 2D array of x coordinates
        y_grid: 2D array of y coordinates
    """
    # Create a grid of positions
    x_range = np.linspace(-1.0, 1.0, resolution)
    y_range = np.linspace(-0.33, 1.33, resolution)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    # Initialize reward component grids
    reward_components = {
        'danger_zone_penalty': np.zeros((resolution, resolution)),
        'danger_avoidance': np.zeros((resolution, resolution)),
        'total_danger_reward': np.zeros((resolution, resolution))
    }
    
    # For each position in the grid, calculate the reward components
    for i in range(resolution):
        for j in range(resolution):
            # Convert grid coordinates to world coordinates
            x = x_grid[i, j]
            y = y_grid[i, j]
            
            # Create a state vector
            state = [
                x,  # x position
                y,  # y position
                velocity[0],  # vx
                velocity[1],  # vy
                angle,  # angle
                angular_velocity,  # angular velocity
                legs_contact[0],  # left leg contact
                legs_contact[1],  # right leg contact
                0,  # left danger zone distance (will be calculated)
                0,  # right danger zone distance (will be calculated)
                0,  # top danger zone distance (will be calculated)
                0,  # bottom danger zone distance (will be calculated)
            ]

            W = VIEWPORT_W / SCALE
            H = VIEWPORT_H / SCALE
            helipad_y = H / 4
            
            # Calculate danger zone distances
            # Convert normalized coordinates to world coordinates
            pos_x = x * (VIEWPORT_W / SCALE / 2) + VIEWPORT_W / SCALE / 2
            pos_y = y * (VIEWPORT_H / SCALE / 2) + (helipad_y + LEG_DOWN / SCALE)
            
            # Create a position object with x and y attributes
            class Pos:
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
            
            pos = Pos(pos_x, pos_y)
            
            # Calculate danger zone distances
            left_dist, right_dist, top_dist, bottom_dist = env._measure_danger_zone_distance(pos)
            state[8] = left_dist
            state[9] = right_dist
            state[10] = top_dist
            state[11] = bottom_dist
            
            # Calculate danger zone penalty
            danger_zone_penalty = -10 * (
                max(0.2 - state[8], 0) +  # left danger zone
                max(0.2 - state[9], 0) +  # right danger zone
                max(0.2 - state[10], 0) + # top danger zone
                max(0.2 - state[11], 0)   # bottom danger zone
            )
            
            # Calculate danger avoidance reward
            # This is a simplified version since we don't have previous state
            # In the actual environment, this would be calculated based on the previous state
            in_danger_zone = state[8] < 0 or state[9] < 0 or state[10] < 0 or state[11] < 0
            
            # For visualization purposes, we'll calculate a proxy for danger avoidance
            # based on the current state only
            danger_avoidance_reward = 0
            if not in_danger_zone:
                # Reward for being far from danger zones
                danger_avoidance_reward = 0.1 * (
                    max(0, state[8]) +  # left danger zone
                    max(0, state[9]) +  # right danger zone
                    max(0, state[10]) + # top danger zone
                    max(0, state[11])   # bottom danger zone
                )
            
            # Store individual reward components
            reward_components['danger_zone_penalty'][i, j] = danger_zone_penalty
            reward_components['danger_avoidance'][i, j] = danger_avoidance_reward
            
            # Calculate total danger reward
            total_danger_reward = danger_zone_penalty + danger_avoidance_reward
            
            # Store the total danger reward
            reward_components['total_danger_reward'][i, j] = total_danger_reward
    
    return reward_components, x_grid, y_grid

def plot_danger_zone_reward_heatmap(env, resolution=50, angle=0.0, velocity=(0, 0), angular_velocity=0.0, legs_contact=(0, 0)):
    """
    Plot heatmaps of danger zone reward components for the Lunar Lander environment.
    
    Args:
        env: The Lunar Lander environment
        resolution: Number of points in each dimension of the grid
        angle: Fixed angle for the lander
        velocity: Fixed velocity (vx, vy) for the lander
        angular_velocity: Fixed angular velocity for the lander
        legs_contact: Fixed legs contact state (left, right)
    """
    # Create the reward component heatmaps
    reward_components, x_grid, y_grid = create_danger_zone_reward_heatmap(
        env, resolution, angle, velocity, angular_velocity, legs_contact
    )
    
    # Create a figure with subplots for each reward component
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot each reward component
    for i, (component, values) in enumerate(reward_components.items()):
        ax = axes[i]
        
        # Find the maximum absolute value for symmetric normalization
        vmax = max(abs(np.min(values)), abs(np.max(values)))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        # Plot the heatmap with centered colormap
        im = ax.pcolormesh(x_grid, y_grid, values, cmap='RdYlGn', norm=norm, shading='auto')
        plt.colorbar(im, ax=ax, label=f'{component.replace("_", " ").capitalize()}')
        
        # Plot the danger zones
        for zone in env.danger_zones:
            x_min, x_max = zone[0]
            y_min, y_max = zone[1]
            width = x_max - x_min
            height = y_max - y_min
            rect = Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
            ax.add_patch(rect)
        
        # Plot the landing pad
        pad_x_min = (env.helipad_x1 - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        pad_x_max = (env.helipad_x2 - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        pad_y = 0  # The landing pad is at y=0 in normalized coordinates
        pad_width = pad_x_max - pad_x_min
        pad_height = 0.05  # Arbitrary height for visualization
        rect = Rectangle((pad_x_min, pad_y - pad_height/2), pad_width, pad_height, linewidth=1, edgecolor='b', facecolor='b', alpha=0.5)
        ax.add_patch(rect)
        
        # Set the title and labels
        ax.set_title(f'{component.replace("_", " ").capitalize()}')
        ax.set_xlabel('X Position (normalized)')
        ax.set_ylabel('Y Position (normalized)')
        
        # Set the axis limits
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.33, 1.33)
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set the main title
    fig.suptitle(f'Danger Zone Rewards (angle={angle:.2f}, v=({velocity[0]:.2f}, {velocity[1]:.2f}), ω={angular_velocity:.2f}, legs={legs_contact})', fontsize=16)
    
    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    # Save the figure
    plt.savefig(f'danger_zone_reward_heatmap_angle{angle:.2f}_v{velocity[0]:.2f}_{velocity[1]:.2f}_w{angular_velocity:.2f}_legs{legs_contact[0]}{legs_contact[1]}.png', dpi=300)
    plt.close()

def main():
    # Create the environment
    env = DangerZoneLunarLander(render_mode=None)
    env.reset()
    
    # Create heatmaps for different conditions
    
    # Default conditions (no velocity, no angle, no legs contact)
    plot_danger_zone_reward_heatmap(env, resolution=100)
    
    # With legs contact
    plot_danger_zone_reward_heatmap(env, resolution=100, legs_contact=(1, 1))
    
    # With some velocity
    plot_danger_zone_reward_heatmap(env, resolution=100, velocity=(0.1, 0.1))
    
    # With some angle
    plot_danger_zone_reward_heatmap(env, resolution=100, angle=0.2)
    
    # With some angular velocity
    plot_danger_zone_reward_heatmap(env, resolution=100, angular_velocity=0.1)
    
    # With legs contact and some velocity
    plot_danger_zone_reward_heatmap(env, resolution=100, legs_contact=(1, 1), velocity=(0.05, 0.05))
    
    # With legs contact and some angle
    plot_danger_zone_reward_heatmap(env, resolution=100, legs_contact=(1, 1), angle=0.1)
    
    print("Danger zone reward heatmaps have been generated and saved as PNG files.")

if __name__ == "__main__":
    main() 