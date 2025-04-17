import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Dict, Any
import matplotlib.patches as mpatches

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def load_trajectory_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load trajectory data from a directory"""
    data_path = Path(data_dir) / "all_episodes.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data

def visualize_trajectory(trajectory_data: List[Dict[str, Any]], episode_idx: int, output_dir: str, policy_name: str = ""):
    """Visualize a single trajectory with emphasis on key time points"""
    # Create directory for plots if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from trajectory
    steps = list(range(len(trajectory_data)))
    agent_actions_type = [step['agent_action_type'] for step in trajectory_data]
    agent_actions = [step['agent_action'] for step in trajectory_data]
    agent_actions_length = [step['agent_action_length'] for step in trajectory_data]
    human_actions = [step['human_action'] for step in trajectory_data]
    overwritten = [step['overwritten'] for step in trajectory_data]
    rewards = [step['reward'] for step in trajectory_data]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot actions
    ax1.set_title(f'Notifications and Human Actions Over Time - {policy_name} (Episode {episode_idx})')
    ax1.set_ylabel('Value')
    
    # Plot human actions
    human_actions_steps = [i for i, h in enumerate(human_actions) if h is not None]
    human_actions_values = [human_actions[i] for i in human_actions_steps]
    ax1.scatter(human_actions_steps, human_actions_values, color='purple', marker='.', label='Human Actions', alpha=0.7)
    
    # Plot overwritten actions
    overwritten_steps = [i for i, is_overwritten in enumerate(overwritten) if is_overwritten == 1]
    overwritten_actions = [human_actions[i] for i in overwritten_steps]
    ax1.scatter(overwritten_steps, overwritten_actions, facecolors='none', edgecolors='red', marker='o', label='Human Action (Overwrite)', alpha=0.7)
    
    # Plot agent actions by type and length
    no_op_steps = [i for i, t in enumerate(agent_actions_type) if t == 0]
    no_op_actions = [agent_actions[i] for i in no_op_steps]
    ax1.scatter(no_op_steps, no_op_actions, color='black', marker='x', label='No-Op', alpha=0.7)

    cont_steps = [i for i, t in enumerate(agent_actions_type) if t == 1]
    cont_actions = [0.5 for i in cont_steps]
    ax1.scatter(cont_steps, cont_actions, color='blue', marker='x', label='Cont.', alpha=0.7)
    
    # Plot notifications with different colors based on length
    noti_l3_steps = [i for i, l in enumerate(agent_actions_length) if l == 2]
    noti_l3_actions = [agent_actions[i] for i in noti_l3_steps]
    ax1.scatter(noti_l3_steps, noti_l3_actions, color='red', marker='x', label='l=2', alpha=0.7)
    
    noti_l4_steps = [i for i, l in enumerate(agent_actions_length) if l == 3]
    noti_l4_actions = [agent_actions[i] for i in noti_l4_steps]
    ax1.scatter(noti_l4_steps, noti_l4_actions, color='orange', marker='x', label='l=3', alpha=0.7)
    
    noti_l5_steps = [i for i, l in enumerate(agent_actions_length) if l == 4]
    noti_l5_actions = [agent_actions[i] for i in noti_l5_steps]
    ax1.scatter(noti_l5_steps, noti_l5_actions, color='yellow', marker='x', label='l=4', alpha=0.7)
    
    noti_l6_steps = [i for i, l in enumerate(agent_actions_length) if l == 5]
    noti_l6_actions = [agent_actions[i] for i in noti_l6_steps]
    ax1.scatter(noti_l6_steps, noti_l6_actions, color='green', marker='x', label='l=5', alpha=0.7)
    
    # Set y-axis limits to match the image style
    ax1.set_ylim(-1, 3.5)
    
    # Move legend to bottom of figure
    ax1.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=8)

    ax1.grid(True, alpha=0.3)
    
    # Plot rewards
    ax2.set_title('Rewards')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.plot(steps, rewards, 'k-', label='Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to file
    plot_path = f"{output_dir}/episode_{episode_idx}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def visualize_comparison(trajectories: List[Dict[str, Any]], episode_indices: List[int], policy_names: List[str], output_dir: str):
    """Create a comparison visualization of trajectories from different policies"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with subplots for each episode
    fig, axes = plt.subplots(len(episode_indices), 2, figsize=(15, 5 * len(episode_indices)), sharex=True)
    if len(episode_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (episode_idx, policy_name, trajectory_data) in enumerate(zip(episode_indices, policy_names, trajectories)):
        # Extract data from trajectory
        steps = list(range(len(trajectory_data)))
        agent_actions_type = [step['agent_action_type'] for step in trajectory_data]
        agent_actions = [step['agent_action'] for step in trajectory_data]
        agent_actions_length = [step['agent_action_length'] for step in trajectory_data]
        human_actions = [step['human_action'] for step in trajectory_data]
        overwritten = [step['overwritten'] for step in trajectory_data]
        rewards = [step['reward'] for step in trajectory_data]
        
        # Plot actions
        axes[i, 0].set_title(f'Episode {episode_idx} - {policy_name}')
        axes[i, 0].set_ylabel('Value')
        
        # Plot human actions
        human_actions_steps = [i for i, h in enumerate(human_actions) if h is not None]
        human_actions_values = [human_actions[i] for i in human_actions_steps]
        axes[i, 0].scatter(human_actions_steps, human_actions_values, color='purple', marker='.', label='Human Actions', alpha=0.7)
        
        # Plot overwritten actions
        overwritten_steps = [i for i, is_overwritten in enumerate(overwritten) if is_overwritten == 1]
        overwritten_actions = [human_actions[i] for i in overwritten_steps]
        axes[i, 0].scatter(overwritten_steps, overwritten_actions, facecolors='none', edgecolors='red', marker='o', label='Human Action (Overwrite)', alpha=0.7)
        
        # Plot agent actions by type and length
        no_op_steps = [i for i, t in enumerate(agent_actions_type) if t == 0]
        no_op_actions = [agent_actions[i] for i in no_op_steps]
        axes[i, 0].scatter(no_op_steps, no_op_actions, color='black', marker='x', label='No-Op', alpha=0.7)

        cont_steps = [i for i, t in enumerate(agent_actions_type) if t == 1]
        cont_actions = [0.5 for i in cont_steps]
        axes[i, 0].scatter(cont_steps, cont_actions, color='blue', marker='x', label='Cont.', alpha=0.7)
        
        # Plot notifications with different colors based on length
        noti_l3_steps = [i for i, l in enumerate(agent_actions_length) if l == 2]
        noti_l3_actions = [agent_actions[i] for i in noti_l3_steps]
        axes[i, 0].scatter(noti_l3_steps, noti_l3_actions, color='red', marker='x', label='l=2', alpha=0.7)
        
        noti_l4_steps = [i for i, l in enumerate(agent_actions_length) if l == 3]
        noti_l4_actions = [agent_actions[i] for i in noti_l4_steps]
        axes[i, 0].scatter(noti_l4_steps, noti_l4_actions, color='orange', marker='x', label='l=3', alpha=0.7)
        
        noti_l5_steps = [i for i, l in enumerate(agent_actions_length) if l == 4]
        noti_l5_actions = [agent_actions[i] for i in noti_l5_steps]
        axes[i, 0].scatter(noti_l5_steps, noti_l5_actions, color='yellow', marker='x', label='l=4', alpha=0.7)
        
        noti_l6_steps = [i for i, l in enumerate(agent_actions_length) if l == 5]
        noti_l6_actions = [agent_actions[i] for i in noti_l6_steps]
        axes[i, 0].scatter(noti_l6_steps, noti_l6_actions, color='green', marker='x', label='l=5', alpha=0.7)
        
        # Set y-axis limits
        axes[i, 0].set_ylim(-1, 3.5)
        axes[i, 0].legend(loc='upper right')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot rewards
        axes[i, 1].set_title('Rewards')
        axes[i, 1].set_ylabel('Reward')
        axes[i, 1].plot(steps, rewards, 'k-', label='Reward')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    # Set x-label for the last row
    axes[-1, 0].set_xlabel('Step')
    axes[-1, 1].set_xlabel('Step')
    
    plt.tight_layout()
    
    # Save plot to file
    plot_path = f"{output_dir}/comparison_episodes_{'_'.join(map(str, episode_indices))}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def create_highlighted_comparison(trajectories: List[Dict[str, Any]], episode_indices: List[int], policy_names: List[str], output_dir: str):
    """Create a comparison visualization with highlighted key time points"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with subplots for each episode
    fig, axes = plt.subplots(len(episode_indices), 1, figsize=(15, 5 * len(episode_indices)), sharex=True)
    if len(episode_indices) == 1:
        axes = [axes]
    
    for i, (episode_idx, policy_name, trajectory_data) in enumerate(zip(episode_indices, policy_names, trajectories)):
        # Extract data from trajectory
        steps = list(range(len(trajectory_data)))
        agent_actions_type = [step['agent_action_type'] for step in trajectory_data]
        agent_actions = [step['agent_action'] for step in trajectory_data]
        agent_actions_length = [step['agent_action_length'] for step in trajectory_data]
        human_actions = [step['human_action'] for step in trajectory_data]
        overwritten = [step['overwritten'] for step in trajectory_data]
        rewards = [step['reward'] for step in trajectory_data]
        
        # Plot actions
        axes[i].set_title(f'Episode {episode_idx} - {policy_name}')
        axes[i].set_ylabel('Value')
        
        # Plot human actions
        human_actions_steps = [i for i, h in enumerate(human_actions) if h is not None]
        human_actions_values = [human_actions[i] for i in human_actions_steps]
        axes[i].scatter(human_actions_steps, human_actions_values, color='purple', marker='.', label='Human Actions', alpha=0.7)
        
        # Plot overwritten actions
        overwritten_steps = [i for i, is_overwritten in enumerate(overwritten) if is_overwritten == 1]
        overwritten_actions = [human_actions[i] for i in overwritten_steps]
        axes[i].scatter(overwritten_steps, overwritten_actions, facecolors='none', edgecolors='red', marker='o', label='Human Action (Overwrite)', alpha=0.7)
        
        # Plot agent actions by type and length
        no_op_steps = [i for i, t in enumerate(agent_actions_type) if t == 0]
        no_op_actions = [agent_actions[i] for i in no_op_steps]
        axes[i].scatter(no_op_steps, no_op_actions, color='black', marker='x', label='No-Op', alpha=0.7)

        cont_steps = [i for i, t in enumerate(agent_actions_type) if t == 1]
        cont_actions = [0.5 for i in cont_steps]
        axes[i].scatter(cont_steps, cont_actions, color='blue', marker='x', label='Cont.', alpha=0.7)
        
        # Plot notifications with different colors based on length
        noti_l3_steps = [i for i, l in enumerate(agent_actions_length) if l == 2]
        noti_l3_actions = [agent_actions[i] for i in noti_l3_steps]
        axes[i].scatter(noti_l3_steps, noti_l3_actions, color='red', marker='x', label='l=2', alpha=0.7)
        
        noti_l4_steps = [i for i, l in enumerate(agent_actions_length) if l == 3]
        noti_l4_actions = [agent_actions[i] for i in noti_l4_steps]
        axes[i].scatter(noti_l4_steps, noti_l4_actions, color='orange', marker='x', label='l=3', alpha=0.7)
        
        noti_l5_steps = [i for i, l in enumerate(agent_actions_length) if l == 4]
        noti_l5_actions = [agent_actions[i] for i in noti_l5_steps]
        axes[i].scatter(noti_l5_steps, noti_l5_actions, color='yellow', marker='x', label='l=4', alpha=0.7)
        
        noti_l6_steps = [i for i, l in enumerate(agent_actions_length) if l == 5]
        noti_l6_actions = [agent_actions[i] for i in noti_l6_steps]
        axes[i].scatter(noti_l6_steps, noti_l6_actions, color='green', marker='x', label='l=5', alpha=0.7)
        
        # Highlight key time points
        # 1. When notifications are conveyed
        notification_steps = []
        for step_idx, (action_type, action_length) in enumerate(zip(agent_actions_type, agent_actions_length)):
            if action_type > 1:  # Notification
                notification_steps.append(step_idx)
        
        # Add vertical lines for notification steps
        for step in notification_steps:
            axes[i].axvline(x=step, color='green', linestyle='--', alpha=0.3)
        
        # 2. When human actions are overwritten
        overwrite_start_steps = []
        prev_overwritten = 0
        for step_idx, is_overwritten in enumerate(overwritten):
            if is_overwritten == 1 and prev_overwritten == 0:
                overwrite_start_steps.append(step_idx)
            prev_overwritten = is_overwritten
        
        # Add vertical lines for overwrite start steps
        for step in overwrite_start_steps:
            axes[i].axvline(x=step, color='red', linestyle='--', alpha=0.3)
        
        # Set y-axis limits
        axes[i].set_ylim(-1, 3.5)
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
    
    # Set x-label for the last row
    axes[-1].set_xlabel('Step')
    
    plt.tight_layout()
    
    # Save plot to file
    plot_path = f"{output_dir}/highlighted_comparison_episodes_{'_'.join(map(str, episode_indices))}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def create_notification_overwrite_comparison(trajectories: List[Dict[str, Any]], episode_indices: List[int], policy_names: List[str], output_dir: str):
    """Create a comparison visualization focusing on notification and overwrite timing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with subplots for each episode
    fig, axes = plt.subplots(len(episode_indices), 1, figsize=(15, 5 * len(episode_indices)), sharex=True)
    if len(episode_indices) == 1:
        axes = [axes]
    
    for i, (episode_idx, policy_name, trajectory_data) in enumerate(zip(episode_indices, policy_names, trajectories)):
        # Extract data from trajectory
        steps = list(range(len(trajectory_data)))
        agent_actions_type = [step['agent_action_type'] for step in trajectory_data]
        agent_actions_length = [step['agent_action_length'] for step in trajectory_data]
        overwritten = [step['overwritten'] for step in trajectory_data]
        
        # Create a timeline of events
        timeline = np.zeros(len(steps))
        
        # Mark notification events
        for step_idx, (action_type, action_length) in enumerate(zip(agent_actions_type, agent_actions_length)):
            if action_type > 1:  # Notification
                timeline[step_idx] = 1  # Notification
        
        # Mark overwrite events
        for step_idx, is_overwritten in enumerate(overwritten):
            if is_overwritten == 1:
                timeline[step_idx] = 2  # Overwrite
        
        # Plot timeline
        axes[i].set_title(f'Episode {episode_idx} - {policy_name}')
        axes[i].set_ylabel('Event Type')
        
        # Create a colormap for the timeline
        cmap = plt.cm.get_cmap('viridis', 3)
        
        # Plot the timeline as a heatmap
        axes[i].imshow(timeline.reshape(1, -1), aspect='auto', cmap=cmap, vmin=0, vmax=2)
        
        # Add custom ticks
        axes[i].set_yticks([0])
        axes[i].set_yticklabels(['Events'])
        
        # Add custom colorbar
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, vmin=0, vmax=2), ax=axes[i], ticks=[0.5, 1.5])
        cbar.set_ticklabels(['No Event', 'Notification', 'Overwrite'])
        
        # Add grid
        axes[i].grid(True, alpha=0.3)
    
    # Set x-label for the last row
    axes[-1].set_xlabel('Step')
    
    plt.tight_layout()
    
    # Save plot to file
    plot_path = f"{output_dir}/notification_overwrite_timeline_episodes_{'_'.join(map(str, episode_indices))}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description='Visualize trajectories from collected data')
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True, help='Directories containing trajectory data')
    parser.add_argument('--policy_names', type=str, nargs='+', help='Names of policies (default: directory names)')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to visualize per policy')
    parser.add_argument('--output_dir', type=str, default='plots/trajectories', help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trajectory data from each directory
    all_data = []
    for data_dir in args.data_dirs:
        data = load_trajectory_data(data_dir)
        all_data.append(data)
        print(f"Loaded {len(data)} episodes from {data_dir}")
    
    # Use directory names as policy names if not provided
    if args.policy_names is None:
        args.policy_names = [Path(data_dir).name for data_dir in args.data_dirs]
    
    # Ensure we have the same number of policy names as data directories
    if len(args.policy_names) != len(args.data_dirs):
        raise ValueError(f"Number of policy names ({len(args.policy_names)}) must match number of data directories ({len(args.data_dirs)})")
    
    # Randomly select episodes to visualize
    selected_episodes = []
    for i, data in enumerate(all_data):
        if len(data) > args.num_episodes:
            episodes = random.sample(range(len(data)), args.num_episodes)
        else:
            episodes = list(range(len(data)))
        
        selected_episodes.append(episodes)
        print(f"Selected episodes {episodes} for policy {args.policy_names[i]}")
    
    # Visualize individual episodes
    for i, (data, policy_name, episodes) in enumerate(zip(all_data, args.policy_names, selected_episodes)):
        policy_dir = os.path.join(args.output_dir, policy_name)
        os.makedirs(policy_dir, exist_ok=True)
        
        for episode_idx in episodes:
            trajectory_data = data[episode_idx]['trajectory']
            plot_path = visualize_trajectory(trajectory_data, episode_idx, policy_dir, policy_name)
            print(f"Saved visualization for episode {episode_idx} to {plot_path}")
    
    # If we have multiple policies, create comparison visualizations
    if len(all_data) > 1:
        # Create a comparison for each selected episode
        for i in range(min(len(episodes) for episodes in selected_episodes)):
            episode_indices = [episodes[i] for episodes in selected_episodes]
            trajectories = [data[episode_idx]['trajectory'] for data, episode_idx in zip(all_data, episode_indices)]
            
            # Create comparison visualization
            plot_path = visualize_comparison(trajectories, episode_indices, args.policy_names, args.output_dir)
            print(f"Saved comparison visualization for episodes {episode_indices} to {plot_path}")
            
            # Create highlighted comparison visualization
            plot_path = create_highlighted_comparison(trajectories, episode_indices, args.policy_names, args.output_dir)
            print(f"Saved highlighted comparison visualization for episodes {episode_indices} to {plot_path}")
            
            # Create notification-overwrite timeline comparison
            plot_path = create_notification_overwrite_comparison(trajectories, episode_indices, args.policy_names, args.output_dir)
            print(f"Saved notification-overwrite timeline comparison for episodes {episode_indices} to {plot_path}")
    
    print(f"Visualization complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main() 