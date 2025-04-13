import argparse
import numpy as np
import gymnasium_envs
import gymnasium as gym
import torch
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)

from agents.heuristic import HeuristicAgent
from agents.humans import HumanAgent
from configs.args import Args

def plot_optimal_path(path, step, positions, danger_zones):
    """Plot a single optimal path with the current state"""
    plt.figure(figsize=(10, 10))
    
    # Plot current trajectory up to this point
    current_positions = positions[:step+1]
    plt.plot(current_positions[:, 0], current_positions[:, 1], 'b-', label='Current Trajectory')
    plt.plot(current_positions[0, 0], current_positions[0, 1], 'go', label='Start')
    plt.plot(current_positions[-1, 0], current_positions[-1, 1], 'bo', label='Current Position')
    
    # Plot optimal path
    path_array = np.array(path)
    plt.plot(path_array[:, 0], path_array[:, 1], 'g--', label='Optimal Path')
    plt.plot(path_array[0, 0], path_array[0, 1], 'g.', label='Path Start')
    plt.plot(path_array[-1, 0], path_array[-1, 1], 'r.', label='Path End')
    
    # Add danger zones
    for zone in danger_zones:
        rect = Rectangle((zone['x'] - zone['width']/2, zone['y'] - zone['height']/2),
                        zone['width'], zone['height'],
                        facecolor='red', alpha=0.3)
        plt.gca().add_patch(rect)
    
    # Add landing pad
    landing_pad = Rectangle((-0.1, -0.1), 0.2, 0.02, facecolor='green', alpha=0.5)
    plt.gca().add_patch(landing_pad)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Optimal Path at Step {step}')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig(f'optimal_path_step_{step}.png')
    plt.close()

def post_analysis(trajectory_data):
    """Plot the lunar lander trajectory and actions"""
    # Extract data
    positions = np.array([d['position'] for d in trajectory_data])
    human_actions = np.array([d['human_action'] for d in trajectory_data])
    notifications = np.array([d['notification'] for d in trajectory_data])
    
    # Define danger zones
    danger_zones = [
        {'x': -1.0, 'y': 0.0, 'width': 0.2, 'height': 0.2},  # Left
        {'x': 1.0, 'y': 0.0, 'width': 0.2, 'height': 0.2},   # Right
        {'x': 0.0, 'y': 1.0, 'width': 0.2, 'height': 0.2},   # Top
        {'x': 0.0, 'y': -1.0, 'width': 0.2, 'height': 0.2}   # Bottom
    ]
    
    # Track unique optimal paths
    seen_paths = set()
    path_updates = []
    
    # Find steps where optimal path changes
    for i, data in enumerate(trajectory_data):
        if 'optimal_path' in data and data['optimal_path'] is not None:
            path_tuple = tuple(map(tuple, data['optimal_path']))  # Convert to hashable type
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                path_updates.append((i, data['optimal_path']))
    
    # Plot each unique optimal path
    for step, path in path_updates:
        plot_optimal_path(path, step, positions, danger_zones)
    
    # Create main analysis figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot trajectory
    ax1 = fig.add_subplot(221)
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', label='Trajectory')
    ax1.plot(positions[0, 0], positions[0, 1], 'go', label='Start')
    ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', label='End')
    
    # Add danger zones
    for zone in danger_zones:
        rect = Rectangle((zone['x'] - zone['width']/2, zone['y'] - zone['height']/2),
                        zone['width'], zone['height'],
                        facecolor='red', alpha=0.3)
        ax1.add_patch(rect)
    
    # Add landing pad
    landing_pad = Rectangle((-0.1, -0.1), 0.2, 0.02, facecolor='green', alpha=0.5)
    ax1.add_patch(landing_pad)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Lunar Lander Trajectory')
    ax1.grid(True)
    ax1.legend()
    
    # Plot human actions
    ax2 = fig.add_subplot(222)
    action_names = ['No Action', 'Left', 'Main', 'Right']
    for i in range(4):
        mask = human_actions == i
        if np.any(mask):
            ax2.plot(np.where(mask)[0], [i] * np.sum(mask), 'bo', label=action_names[i])
    ax2.set_ylabel('Action')
    ax2.set_xlabel('Step')
    ax2.set_title('Human Actions')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(action_names)
    ax2.grid(True)
    
    # Plot notifications
    ax3 = fig.add_subplot(223)
    notification_types = ['No Notification', 'Continue Previous', 'New Notification']
    for i in range(3):
        mask = notifications[:, 0] == i
        if np.any(mask):
            ax3.plot(np.where(mask)[0], [i] * np.sum(mask), 'ro', label=notification_types[i])
    ax3.set_ylabel('Notification Type')
    ax3.set_xlabel('Step')
    ax3.set_title('Notifications')
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(notification_types)
    ax3.grid(True)
    
    # Plot notification actions
    ax4 = fig.add_subplot(224)
    action_names = ['No Action', 'Left', 'Main', 'Right']
    for i in range(4):
        mask = notifications[:, 1] == i
        if np.any(mask):
            ax4.plot(np.where(mask)[0], [i] * np.sum(mask), 'go', label=action_names[i])
    ax4.set_ylabel('Action')
    ax4.set_xlabel('Step')
    ax4.set_title('Notification Actions')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(action_names)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('lunar_lander_analysis.png')
    plt.close()
    
    print(f"Created {len(path_updates)} optimal path plots")
    for step, _ in path_updates:
        print(f"- optimal_path_step_{step}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="lunarlander")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--human_agent_type", type=str, default="mlp", help="Type of human agent to use")
    parser.add_argument("--human_agent_path", type=str, default=None, help="Path to the human agent model")
    args = parser.parse_args()
    
    # Create a minimal args object for the agents
    agent_args = Args()
    agent_args.human_agent_type = args.human_agent_type
    agent_args.human_agent_path = args.human_agent_path
    agent_args.num_envs = 1
    agent_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the environment
    if args.env == "lunarlander":
        env = gym.make("DangerZoneLunarLander", render_mode="rgb_array" if args.render else None)
    elif args.env == "highway":
        env = gym.make("gymnasium_envs/NotiHighway", render_mode="rgb_array" if args.render else None)
    elif args.env == "highway_fast":
        env = gym.make("gymnasium_envs/NotiHighwayFast", render_mode="rgb_array" if args.render else None)
    else:
        raise ValueError(f"Environment {args.env} not found")
    
    # Create vectorized environment for the agents
    envs = gym.vector.SyncVectorEnv([lambda: env])
    
    # Initialize the agents
    heuristic_agent = HeuristicAgent(envs, agent_args).to(agent_args.device)
    human_agent = HumanAgent(envs, agent_args, agent_args.device)
    
    # Reset the environment
    observation, info = env.reset()
    observation = torch.tensor(observation).unsqueeze(0).to(agent_args.device)
    
    # Run the environment
    done = False
    total_reward = 0
    step = 0
    
    # Store trajectory data for analysis
    trajectory_data = []
    
    print("Starting episode with heuristic agent providing notifications...")
    
    while not done:
        # Get notification from heuristic agent
        with torch.no_grad():
            notification, _, _, _ = heuristic_agent.get_action_and_value(observation)
        
        # Get human action based on the notification
        human_action, overwrite_flag = human_agent.get_action(observation, notification.cpu().numpy())
        
        # Combine actions for the environment
        full_action = np.concatenate([notification.cpu().numpy(), human_action.cpu().numpy().reshape(-1, 1), overwrite_flag.reshape(-1, 1)], axis=1)
        
        # Step the environment
        next_observation, reward, terminated, truncated, info = env.step(full_action[0])
        done = np.logical_or(terminated, truncated)
        
        # Store trajectory data
        trajectory_data.append({
            'position': (observation[0, 0].item(), observation[0, 1].item()),
            'human_action': human_action[0].item(),
            'notification': notification[0].cpu().numpy(),
            'overwrite_flag': overwrite_flag[0].item(),
            'optimal_path': heuristic_agent.optimal_path if hasattr(heuristic_agent, 'optimal_path') else None
        })
        
        # Update observation
        observation = torch.tensor(next_observation).unsqueeze(0).to(agent_args.device)
        
        # Accumulate reward
        total_reward += reward
        
        # Print step information
        step += 1
        if step % 10 == 0:
            print(f"Step {step}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            print(f"Notification: {notification.cpu().numpy()[0]}")
            print(f"Human Action: {human_action.cpu().numpy()[0]}")
        
        # Render if requested
        if args.render:
            env.render()
    
    print(f"Episode finished with total reward: {total_reward:.2f}")
    
    # Perform post-analysis
    post_analysis(trajectory_data)
    print("Analysis plot saved as 'lunar_lander_analysis.png'")
    
    env.close()