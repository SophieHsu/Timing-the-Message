import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple
import seaborn as sns
from collections import Counter, defaultdict
import matplotlib.patches as mpatches
from gymnasium_envs.envs.lunar_lander import VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN

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

def analyze_rewards(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze reward statistics across episodes"""
    rewards = [episode['total_reward'] for episode in data]
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    median_reward = np.median(rewards)
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot reward distribution
    sns.histplot(rewards, kde=True)
    plt.axvline(mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    plt.axvline(median_reward, color='g', linestyle='--', label=f'Median: {median_reward:.2f}')
    
    plt.title(f'Reward Distribution - {policy_name}')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/reward_distribution_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Mean', 'Std', 'Min', 'Max', 'Median'],
        'Value': [mean_reward, std_reward, min_reward, max_reward, median_reward]
    })
    
    # Save the summary
    summary.to_csv(f"{output_dir}/reward_summary_{policy_name}.csv", index=False)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'median_reward': median_reward
    }

def analyze_episode_lengths(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze episode length statistics"""
    lengths = [episode['num_steps'] for episode in data]
    
    # Calculate statistics
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    median_length = np.median(lengths)
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot length distribution
    sns.histplot(lengths, kde=True)
    plt.axvline(mean_length, color='r', linestyle='--', label=f'Mean: {mean_length:.2f}')
    plt.axvline(median_length, color='g', linestyle='--', label=f'Median: {median_length:.2f}')
    
    plt.title(f'Episode Length Distribution - {policy_name}')
    plt.xlabel('Number of Steps')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/episode_length_distribution_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Mean', 'Std', 'Min', 'Max', 'Median'],
        'Value': [mean_length, std_length, min_length, max_length, median_length]
    })
    
    # Save the summary
    summary.to_csv(f"{output_dir}/episode_length_summary_{policy_name}.csv", index=False)
    
    return {
        'mean_length': mean_length,
        'std_length': std_length,
        'min_length': min_length,
        'max_length': max_length,
        'median_length': median_length
    }

def analyze_action_distribution(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze the distribution of actions taken by the agent and human"""
    agent_action_types = []
    agent_actions = []
    agent_action_lengths = []
    human_actions = []
    overwritten = []
    
    for episode in data:
        for step in episode['trajectory']:
            agent_action_types.append(step['agent_action_type'])
            agent_actions.append(step['agent_action'])
            agent_action_lengths.append(step['agent_action_length'])
            human_actions.append(step['human_action'])
            overwritten.append(step['overwritten'])
    
    # Count frequencies
    agent_action_type_counts = Counter(agent_action_types)
    agent_action_counts = Counter(agent_actions)
    agent_action_length_counts = Counter(agent_action_lengths)
    human_action_counts = Counter(human_actions)
    overwritten_counts = Counter(overwritten)
    
    # Create figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot agent action type distribution
    ax = axes[0, 0]
    action_type_labels = ['No-op', 'Continue', 'Notification']
    action_type_values = [agent_action_type_counts[i] for i in range(3)]
    ax.bar(action_type_labels, action_type_values)
    ax.set_title('Agent Action Type Distribution')
    ax.set_ylabel('Count')
    
    # Plot agent action distribution
    ax = axes[0, 1]
    action_labels = [str(i) for i in range(4)]
    action_values = [agent_action_counts[i] for i in range(4)]
    ax.bar(action_labels, action_values)
    ax.set_title('Agent Action Distribution')
    ax.set_ylabel('Count')
    
    # Plot agent action length distribution
    ax = axes[1, 0]
    length_labels = [str(i) for i in range(6)]
    length_values = [agent_action_length_counts[i] for i in range(6)]
    ax.bar(length_labels, length_values)
    ax.set_title('Agent Action Length Distribution')
    ax.set_xlabel('Length')
    ax.set_ylabel('Count')
    
    # Plot human action distribution
    ax = axes[1, 1]
    human_labels = [str(i) for i in range(4)]
    human_values = [human_action_counts[i] for i in range(4)]
    ax.bar(human_labels, human_values)
    ax.set_title('Human Action Distribution')
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/action_distribution_{policy_name}.png")
    plt.close()
    
    # Create a separate plot for overwritten actions
    plt.figure(figsize=(8, 6))
    overwrite_labels = ['Not Overwritten', 'Overwritten']
    overwrite_values = [overwritten_counts[0], overwritten_counts[1]]
    plt.bar(overwrite_labels, overwrite_values)
    plt.title('Action Overwrite Distribution')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/overwrite_distribution_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Action Type': ['No-op', 'Continue', 'Notification'],
        'Count': [agent_action_type_counts[0], agent_action_type_counts[1], agent_action_type_counts[2]]
    })
    
    # Save the summary
    summary.to_csv(f"{output_dir}/action_type_summary_{policy_name}.csv", index=False)
    
    return {
        'agent_action_type_counts': agent_action_type_counts,
        'agent_action_counts': agent_action_counts,
        'agent_action_length_counts': agent_action_length_counts,
        'human_action_counts': human_action_counts,
        'overwritten_counts': overwritten_counts
    }

def analyze_notifications(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze notification patterns and statistics."""
    # Initialize lists to store notification data
    notification_steps = []
    notification_actions = []
    notification_lengths = []
    human_actions_after_notification = []
    overwritten_after_notification = []
    
    # Analyze each trajectory
    for episode_idx, episode in enumerate(data):
        trajectory = episode['trajectory']
        
        for i, step in enumerate(trajectory):
            if step['agent_action_type'] == 2:  # Notification
                notification_steps.append(i)
                notification_actions.append(step['agent_action'])
                notification_lengths.append(step['agent_action_length'])
                
                # Look at human action after notification
                if i + 1 < len(trajectory):
                    human_actions_after_notification.append(trajectory[i+1]['human_action'])
                    overwritten_after_notification.append(trajectory[i+1]['overwritten'])
    
    # Calculate statistics
    total_notifications = len(notification_steps)
    avg_notification_length = np.mean(notification_lengths) if notification_lengths else 0
    std_notification_length = np.std(notification_lengths) if notification_lengths else 0
    
    # Calculate overwrite rate
    if overwritten_after_notification:
        overwrite_count = sum(1 for o in overwritten_after_notification if o == 1)
        overwrite_rate = overwrite_count / len(overwritten_after_notification)
    else:
        overwrite_rate = 0
    
    # Create a figure for notification timing
    plt.figure(figsize=(10, 6))
    plt.hist(notification_steps, bins=20, alpha=0.7)
    plt.title(f'Notification Timing Distribution - {policy_name}')
    plt.xlabel('Step in Episode')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/notification_timing_{policy_name}.png")
    plt.close()
    
    # Create a figure for notification length distribution
    plt.figure(figsize=(10, 6))
    if notification_lengths:
        plt.hist(notification_lengths, bins=range(min(notification_lengths), max(notification_lengths) + 2), alpha=0.7)
    plt.title(f'Notification Length Distribution - {policy_name}')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/notification_length_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Total Notifications', 'Average Notification Length', 'Std Dev Notification Length', 'Overwrite Rate After Notifications'],
        'Value': [total_notifications, avg_notification_length, std_notification_length, overwrite_rate]
    })
    
    # Save the summary
    summary.to_csv(f"{output_dir}/notification_summary_{policy_name}.csv", index=False)
    
    return {
        'total_notifications': total_notifications,
        'avg_notification_length': avg_notification_length,
        'std_notification_length': std_notification_length,
        'overwrite_rate': overwrite_rate
    }

def analyze_danger_zone_interactions(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze interactions with danger zones or road hazards"""
    # Check if the data contains danger zone information
    if not data or 'trajectory' not in data[0] or not data[0]['trajectory']:
        print("No trajectory data found for danger zone analysis")
        return {}
    
    # Check if the first trajectory step has danger zone information
    first_step = data[0]['trajectory'][0]
    has_danger_zones = 'distance_to_danger' in first_step
    
    if not has_danger_zones:
        print("No danger zone information found in trajectory data")
        return {}
    
    danger_zone_distances = {
        'left': [],
        'right': [],
        'top': [],
        'bottom': []
    }
    
    danger_zone_notifications = {
        'left': [],
        'right': [],
        'top': [],
        'bottom': []
    }
    
    # Track episodes with danger zone entry
    episodes_with_entry = 0
    steps_to_entry = []
    
    for episode in data:
        trajectory = episode['trajectory']
        entry_occurred = False
        
        for i, step in enumerate(trajectory):
            # Record distances to danger zones
            for direction in danger_zone_distances:
                if direction in step['distance_to_danger'] and step['distance_to_danger'][direction] is not None:
                    danger_zone_distances[direction].append(step['distance_to_danger'][direction])
            
            # Check if there's a notification and record the distances at that time
            if step['agent_action_type'] == 2:  # Notification
                for direction in danger_zone_notifications:
                    if direction in step['distance_to_danger'] and step['distance_to_danger'][direction] is not None:
                        danger_zone_notifications[direction].append(step['distance_to_danger'][direction])
            
            # Check if entry into any danger zone occurred
            if not entry_occurred:
                for direction, distance in step['distance_to_danger'].items():
                    if distance is not None and distance <= 0:  # Entry into danger zone
                        entry_occurred = True
                        episodes_with_entry += 1
                        steps_to_entry.append(i)
                        break
    
    # Calculate entry rate
    total_episodes = len(data)
    entry_rate = episodes_with_entry / total_episodes if total_episodes > 0 else 0
    
    # Calculate average steps to entry
    avg_steps_to_entry = np.mean(steps_to_entry) if steps_to_entry else 0
    
    # Create a figure for danger zone distance distribution
    plt.figure(figsize=(12, 8))
    
    for i, (direction, distances) in enumerate(danger_zone_distances.items()):
        if distances:  # Only create subplot if we have data
            plt.subplot(2, 2, i+1)
            sns.histplot(distances, kde=True)
            plt.title(f'{direction.capitalize()} Danger Zone Distance')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/danger_zone_distances_{policy_name}.png")
    plt.close()
    
    # Create a figure for danger zone distances during notifications
    plt.figure(figsize=(12, 8))
    
    for i, (direction, distances) in enumerate(danger_zone_notifications.items()):
        plt.subplot(2, 2, i+1)
        if distances:
            sns.histplot(distances, kde=True)
            plt.title(f'{direction.capitalize()} Danger Zone Distance During Notifications')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No notifications', ha='center', va='center')
            plt.title(f'{direction.capitalize()} Danger Zone Distance During Notifications')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/danger_zone_distances_notifications_{policy_name}.png")
    plt.close()
    
    # Calculate statistics
    danger_zone_stats = {}
    for direction, distances in danger_zone_distances.items():
        if distances:  # Only calculate stats if we have data
            danger_zone_stats[direction] = {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            }
    
    # Create a summary DataFrame
    summary_data = []
    for direction, stats in danger_zone_stats.items():
        for stat, value in stats.items():
            summary_data.append({
                'Direction': direction,
                'Statistic': stat,
                'Value': value
            })
    
    # Add entry rate and average steps to entry
    summary_data.append({
        'Direction': 'All',
        'Statistic': 'Entry Rate',
        'Value': entry_rate
    })
    
    summary_data.append({
        'Direction': 'All',
        'Statistic': 'Average Steps to Entry',
        'Value': avg_steps_to_entry
    })
    
    summary = pd.DataFrame(summary_data)
    
    # Save the summary
    summary.to_csv(f"{output_dir}/danger_zone_summary_{policy_name}.csv", index=False)
    
    return {
        'stats': danger_zone_stats,
        'entry_rate': entry_rate,
        'avg_steps_to_entry': avg_steps_to_entry
    }

def analyze_trajectory_patterns(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze trajectory patterns and visualize them"""
    # Check if the data contains trajectory information
    if not data or 'trajectory' not in data[0] or not data[0]['trajectory']:
        print("No trajectory data found for trajectory pattern analysis")
        return {}
    
    # Extract trajectory data
    trajectories = []
    for episode in data:
        trajectory = []
        for step in episode['trajectory']:
            # Extract position information if available
            position = {}
            
            # For Lunar Lander, positions are in the observation
            if 'observation' in step and len(step['observation']) >= 2:
                position['x'] = step['observation'][0]
                position['y'] = step['observation'][1]
            
            # For Highway environment, check if vehicle position is available
            elif 'info' in step and 'vehicle_position' in step['info']:
                position['x'] = step['info']['vehicle_position'][0]
                position['y'] = step['info']['vehicle_position'][1]
            
            # Add action information
            action = {
                'agent_action': step['agent_action'],
                'human_action': step['human_action'],
                'overwritten': step['overwritten']
            }
            
            # Add position and action to trajectory step
            trajectory_step = {**position, **action}
            trajectory.append(trajectory_step)
        
        trajectories.append(trajectory)
    
    # Check if we have position data
    has_position_data = any('x' in step for trajectory in trajectories for step in trajectory)
    
    if not has_position_data:
        print("No position data found in trajectory data")
        return {}
    
    # Create trajectory visualization
    plt.figure(figsize=(12, 8))
    
    # Plot a subset of trajectories (up to 10) to avoid overcrowding
    num_trajectories = min(10, len(trajectories))
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
    
    for i, trajectory in enumerate(trajectories[:num_trajectories]):
        x_coords = [step.get('x', 0) for step in trajectory if 'x' in step]
        y_coords = [step.get('y', 0) for step in trajectory if 'y' in step]
        
        if x_coords and y_coords:
            plt.plot(x_coords, y_coords, color=colors[i], alpha=0.7, label=f'Trajectory {i+1}')
            
            # Mark start and end points
            plt.scatter(x_coords[0], y_coords[0], color=colors[i], marker='o', s=50)
            plt.scatter(x_coords[-1], y_coords[-1], color=colors[i], marker='x', s=50)
    
    plt.title('Trajectory Visualization')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(f"{output_dir}/trajectory_patterns_{policy_name}.png")
    plt.close()
    
    # Analyze action patterns along trajectories
    action_patterns = []
    for trajectory in trajectories:
        pattern = []
        for step in trajectory:
            if 'agent_action' in step and 'human_action' in step:
                pattern.append((step['agent_action'], step['human_action'], step['overwritten']))
        action_patterns.append(pattern)
    
    # Count common action patterns
    pattern_counts = Counter([tuple(pattern) for pattern in action_patterns])
    
    # Create a summary of the most common patterns
    common_patterns = pattern_counts.most_common(5)
    
    # Create a summary DataFrame
    summary_data = []
    for pattern, count in common_patterns:
        summary_data.append({
            'Pattern': str(pattern),
            'Count': count,
            'Percentage': count / len(action_patterns) * 100
        })
    
    summary = pd.DataFrame(summary_data)
    
    # Save the summary
    summary.to_csv(f"{output_dir}/trajectory_pattern_summary_{policy_name}.csv", index=False)
    
    return {
        'common_patterns': common_patterns,
        'num_trajectories': len(trajectories)
    }

def analyze_success_rate(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze success rate and related statistics"""
    # Check if the data contains success information
    if not data or 'trajectory' not in data[0] or not data[0]['trajectory']:
        print("No trajectory data found for success rate analysis")
        return {}
    
    # Initialize counters
    successful_episodes = 0
    steps_to_success = []
    
    # Check if the data contains danger zone information
    first_step = data[0]['trajectory'][0]
    has_danger_zones = 'distance_to_danger' in first_step
    
    # Check each episode for success
    for episode in data:
        trajectory = episode['trajectory']
        success = False
        steps = 0
        entered_danger_zone = False
        
        # Check if the episode entered a danger zone
        if has_danger_zones:
            for step in trajectory:
                for direction, distance in step['distance_to_danger'].items():
                    if distance is not None and distance <= 0:  # Entry into danger zone
                        entered_danger_zone = True
                        break
                if entered_danger_zone:
                    break
        
        # Check if the episode was successful
        # For Lunar Lander, check if the lander landed successfully
        if 'info' in trajectory[-1] and 'success' in trajectory[-1]['info']:
            success = trajectory[-1]['info']['success']
        # For Highway environment, check if the vehicle reached the end without crashing
        elif 'info' in trajectory[-1] and 'crashed' in trajectory[-1]['info']:
            success = not trajectory[-1]['info']['crashed']
        
        if success:
            successful_episodes += 1
            # Only count steps to success if the episode didn't enter a danger zone
            if not entered_danger_zone:
                steps_to_success.append(len(trajectory))
    
    # Calculate success rate
    total_episodes = len(data)
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
    
    # Calculate average steps to success
    avg_steps_to_success = np.mean(steps_to_success) if steps_to_success else 0
    
    # Create a figure for success rate
    plt.figure(figsize=(8, 6))
    
    # Create a pie chart
    labels = ['Successful', 'Unsuccessful']
    sizes = [successful_episodes, total_episodes - successful_episodes]
    colors = ['#4CAF50', '#F44336']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Success Rate')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/success_rate_{policy_name}.png")
    plt.close()
    
    # Create a figure for steps to success distribution
    if steps_to_success:
        plt.figure(figsize=(10, 6))
        
        sns.histplot(steps_to_success, kde=True)
        plt.axvline(avg_steps_to_success, color='r', linestyle='--', label=f'Mean: {avg_steps_to_success:.2f}')
        
        plt.title('Steps to Success Distribution')
        plt.xlabel('Number of Steps')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(f"{output_dir}/steps_to_success_{policy_name}.png")
        plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Success Rate', 'Average Steps to Success', 'Total Episodes', 'Successful Episodes'],
        'Value': [success_rate, avg_steps_to_success, total_episodes, successful_episodes]
    })
    
    # Save the summary
    summary.to_csv(f"{output_dir}/success_summary_{policy_name}.csv", index=False)
    
    return {
        'success_rate': success_rate,
        'avg_steps_to_success': avg_steps_to_success,
        'total_episodes': total_episodes,
        'successful_episodes': successful_episodes
    }

def analyze_human_agent_interaction(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze interaction patterns between human and agent"""
    # Extract interaction data
    human_actions = []
    agent_actions = []
    overwritten = []
    notifications = []
    
    for episode in data:
        for step in episode['trajectory']:
            human_actions.append(step['human_action'])
            agent_actions.append(step['agent_action'])
            overwritten.append(step['overwritten'])
            notifications.append(step['agent_action_type'] == 2)
    
    # Calculate agreement rate (when human and agent choose the same action)
    agreement_count = sum(1 for h, a in zip(human_actions, agent_actions) if h == a)
    agreement_rate = agreement_count / len(human_actions) if human_actions else 0
    
    # Calculate overwrite rate
    overwrite_count = sum(1 for o in overwritten if o == 1)
    overwrite_rate = overwrite_count / len(overwritten) if overwritten else 0
    
    # Calculate notification rate
    notification_count = sum(1 for n in notifications if n)
    notification_rate = notification_count / len(notifications) if notifications else 0
    
    # Determine the action space size dynamically
    max_human_action = max(human_actions) if human_actions else 0
    max_agent_action = max(agent_actions) if agent_actions else 0
    action_space_size = max(max_human_action, max_agent_action) + 1
    
    # Create a confusion matrix for human vs agent actions
    confusion_matrix = np.zeros((action_space_size, action_space_size))
    for h, a in zip(human_actions, agent_actions):
        confusion_matrix[h, a] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    
    # Create action labels based on the environment type
    # For Lunar Lander: [0: do nothing, 1: fire left, 2: fire main, 3: fire right]
    # For Highway: [0: lane left, 1: idle, 2: lane right, 3: faster, 4: slower]
    if action_space_size == 4:
        action_labels = ['No-op', 'Left', 'Main', 'Right']
    elif action_space_size == 5:
        action_labels = ['Lane Left', 'Idle', 'Lane Right', 'Faster', 'Slower']
    else:
        action_labels = [str(i) for i in range(action_space_size)]
    
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=action_labels,
                yticklabels=action_labels)
    plt.title(f'Human vs Agent Action Confusion Matrix - {policy_name}')
    plt.xlabel('Agent Action')
    plt.ylabel('Human Action')
    plt.savefig(f"{output_dir}/confusion_matrix_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Agreement Rate', 'Overwrite Rate', 'Notification Rate'],
        'Value': [agreement_rate, overwrite_rate, notification_rate]
    })
    
    # Save the summary
    summary.to_csv(f"{output_dir}/interaction_summary_{policy_name}.csv", index=False)
    
    return {
        'agreement_rate': agreement_rate,
        'overwrite_rate': overwrite_rate,
        'notification_rate': notification_rate,
        'confusion_matrix': confusion_matrix
    }

def analyze_per_trajectory_notifications(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze notification statistics per trajectory"""
    # Initialize lists to store per-trajectory statistics
    notifications_per_trajectory = []
    notification_lengths_per_trajectory = []
    overwrite_rates_per_trajectory = []
    
    # Track notification length frequencies per trajectory
    notification_length_frequencies = []
    
    # Track notification length selection rates per trajectory
    notification_length_selection_rates = []
    
    # Analyze each trajectory
    for episode_idx, episode in enumerate(data):
        trajectory = episode['trajectory']
        
        # Count notifications in this trajectory
        notification_steps = []
        notification_lengths = []
        overwritten_after_notification = []
        
        # Track notification length frequencies for this trajectory
        length_counts = defaultdict(int)
        
        for i, step in enumerate(trajectory):
            if step['agent_action_type'] == 2:  # Notification
                notification_steps.append(i)
                notification_lengths.append(step['agent_action_length'])
                length_counts[step['agent_action_length']] += 1
                
                # Check if the next step was overwritten
                if i + 1 < len(trajectory):
                    overwritten_after_notification.append(trajectory[i+1]['overwritten'])
        
        # Calculate statistics for this trajectory
        num_notifications = len(notification_steps)
        avg_notification_length = np.mean(notification_lengths) if notification_lengths else 0
        
        # Calculate overwrite rate for this trajectory
        if overwritten_after_notification:
            overwrite_count = sum(1 for o in overwritten_after_notification if o == 1)
            overwrite_rate = overwrite_count / len(overwritten_after_notification)
        else:
            overwrite_rate = 0
        
        # Calculate notification length selection rates for this trajectory
        length_selection_rates = {}
        if num_notifications > 0:
            for length, count in length_counts.items():
                length_selection_rates[length] = count / num_notifications
        
        # Store statistics
        notifications_per_trajectory.append(num_notifications)
        notification_lengths_per_trajectory.append(avg_notification_length)
        overwrite_rates_per_trajectory.append(overwrite_rate)
        
        # Store notification length frequencies for this trajectory
        notification_length_frequencies.append(dict(length_counts))
        
        # Store notification length selection rates for this trajectory
        notification_length_selection_rates.append(length_selection_rates)
    
    # Calculate overall statistics
    avg_notifications_per_trajectory = np.mean(notifications_per_trajectory)
    std_notifications_per_trajectory = np.std(notifications_per_trajectory)
    avg_notification_length_per_trajectory = np.mean(notification_lengths_per_trajectory)
    std_notification_length_per_trajectory = np.std(notification_lengths_per_trajectory)
    avg_overwrite_rate_per_trajectory = np.mean(overwrite_rates_per_trajectory)
    std_overwrite_rate_per_trajectory = np.std(overwrite_rates_per_trajectory)
    
    # Calculate average frequency of each notification length across all trajectories
    all_lengths = set()
    for freq_dict in notification_length_frequencies:
        all_lengths.update(freq_dict.keys())
    
    avg_length_frequencies = {}
    for length in sorted(all_lengths):
        # Calculate the average frequency of this length across all trajectories
        frequencies = [freq_dict.get(length, 0) for freq_dict in notification_length_frequencies]
        avg_length_frequencies[length] = np.mean(frequencies)
    
    # Calculate average selection rate of each notification length across all trajectories
    avg_length_selection_rates = {}
    for length in sorted(all_lengths):
        # Calculate the average selection rate of this length across all trajectories
        selection_rates = [rate_dict.get(length, 0) for rate_dict in notification_length_selection_rates]
        avg_length_selection_rates[length] = np.mean(selection_rates)
    
    # Assert that the selection rates sum to 1 (with a small tolerance for floating-point errors)
    total_rate = sum(avg_length_selection_rates.values())
    assert abs(total_rate - 1.0) < 1e-10, f"Notification length selection rates should sum to 1, but sum to {total_rate}"
    
    # Create a figure for notifications per trajectory distribution
    plt.figure(figsize=(10, 6))
    plt.hist(notifications_per_trajectory, bins=range(min(notifications_per_trajectory), max(notifications_per_trajectory) + 2), alpha=0.7)
    plt.axvline(avg_notifications_per_trajectory, color='r', linestyle='--', label=f'Mean: {avg_notifications_per_trajectory:.2f}')
    plt.title(f'Notifications Per Trajectory - {policy_name}')
    plt.xlabel('Number of Notifications')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/notifications_per_trajectory_{policy_name}.png")
    plt.close()
    
    # Create a figure for notification length per trajectory distribution
    plt.figure(figsize=(10, 6))
    plt.hist(notification_lengths_per_trajectory, bins=10, alpha=0.7)
    plt.axvline(avg_notification_length_per_trajectory, color='r', linestyle='--', label=f'Mean: {avg_notification_length_per_trajectory:.2f}')
    plt.title(f'Average Notification Length Per Trajectory - {policy_name}')
    plt.xlabel('Average Notification Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/notification_length_per_trajectory_{policy_name}.png")
    plt.close()
    
    # Create a figure for overwrite rate per trajectory distribution
    plt.figure(figsize=(10, 6))
    plt.hist(overwrite_rates_per_trajectory, bins=10, alpha=0.7)
    plt.axvline(avg_overwrite_rate_per_trajectory, color='r', linestyle='--', label=f'Mean: {avg_overwrite_rate_per_trajectory:.2f}')
    plt.title(f'Overwrite Rate Per Trajectory - {policy_name}')
    plt.xlabel('Overwrite Rate')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/overwrite_rate_per_trajectory_{policy_name}.png")
    plt.close()
    
    # Create a figure for notification length frequency distribution
    plt.figure(figsize=(10, 6))
    lengths = list(avg_length_frequencies.keys())
    frequencies = list(avg_length_frequencies.values())
    plt.bar(lengths, frequencies, alpha=0.7)
    plt.title(f'Average Notification Length Frequency - {policy_name}')
    plt.xlabel('Notification Length')
    plt.ylabel('Average Frequency Per Trajectory')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/notification_length_frequency_{policy_name}.png")
    plt.close()
    
    # Create a figure for notification length selection rate distribution
    plt.figure(figsize=(10, 6))
    lengths = list(avg_length_selection_rates.keys())
    selection_rates = list(avg_length_selection_rates.values())
    plt.bar(lengths, selection_rates, alpha=0.7)
    plt.title(f'Average Notification Length Selection Rate - {policy_name}')
    plt.xlabel('Notification Length')
    plt.ylabel('Selection Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/notification_length_selection_rate_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Metric': [
            'Average Notifications Per Trajectory',
            'Std Dev Notifications Per Trajectory',
            'Average Notification Length Per Trajectory',
            'Std Dev Notification Length Per Trajectory',
            'Average Overwrite Rate Per Trajectory',
            'Std Dev Overwrite Rate Per Trajectory'
        ],
        'Value': [
            avg_notifications_per_trajectory,
            std_notifications_per_trajectory,
            avg_notification_length_per_trajectory,
            std_notification_length_per_trajectory,
            avg_overwrite_rate_per_trajectory,
            std_overwrite_rate_per_trajectory
        ]
    })
    
    # Add notification length frequency data to the summary
    for length, freq in avg_length_frequencies.items():
        summary.loc[len(summary)] = [f'Average Frequency of Length {length}', freq]
    
    # Add notification length selection rate data to the summary
    for length, rate in avg_length_selection_rates.items():
        summary.loc[len(summary)] = [f'Selection Rate of Length {length}', rate]
    
    # Save the summary
    summary.to_csv(f"{output_dir}/per_trajectory_notification_summary_{policy_name}.csv", index=False)
    
    return {
        'avg_notifications_per_trajectory': avg_notifications_per_trajectory,
        'std_notifications_per_trajectory': std_notifications_per_trajectory,
        'avg_notification_length_per_trajectory': avg_notification_length_per_trajectory,
        'std_notification_length_per_trajectory': std_notification_length_per_trajectory,
        'avg_overwrite_rate_per_trajectory': avg_overwrite_rate_per_trajectory,
        'std_overwrite_rate_per_trajectory': std_overwrite_rate_per_trajectory,
        'notification_length_frequencies': avg_length_frequencies,
        'notification_length_selection_rates': avg_length_selection_rates
    }

def analyze_final_notification_distance(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze the distance from danger zones at the time of the final notification before entry"""
    # Initialize lists to store distances
    final_notification_distances = {
        'bottom': []
    }
    
    # Track trajectories where entry occurred
    trajectories_with_entry = 0
    trajectories_with_notification = 0
    
    # Analyze each trajectory
    for episode_idx, episode in enumerate(data):
        trajectory = episode['trajectory']
        
        # Find the step where entry into a danger zone occurred
        entry_step = None
        entry_direction = None
        
        for i, step in enumerate(trajectory):
            # Check if the agent entered the bottom danger zone
            if step['distance_to_danger']['bottom'] <= 0:  # Entry into bottom danger zone
                entry_step = i
                entry_direction = 'bottom'
                break
        
        # If entry occurred, find the last notification before entry
        if entry_step is not None:
            trajectories_with_entry += 1
            last_notification_step = None
            
            # Look for the last notification before entry
            for i in range(entry_step - 1, -1, -1):
                if trajectory[i]['agent_action_type'] == 2:  # Notification
                    last_notification_step = i
                    break
            
            # If a notification was found before entry, record the distances
            if last_notification_step is not None:
                trajectories_with_notification += 1
                final_notification_distances['bottom'].append(
                    trajectory[last_notification_step]['distance_to_danger']['bottom']
                )
    
    # Calculate statistics
    stats = {}
    if final_notification_distances['bottom']:
        stats['bottom'] = {
            'mean': np.mean(final_notification_distances['bottom']),
            'std': np.std(final_notification_distances['bottom']),
            'min': np.min(final_notification_distances['bottom']),
            'max': np.max(final_notification_distances['bottom']),
            'count': len(final_notification_distances['bottom'])
        }
    else:
        stats['bottom'] = {
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'count': 0
        }
    
    # Calculate notification rate before entry
    notification_rate = trajectories_with_notification / trajectories_with_entry if trajectories_with_entry > 0 else 0
    
    # Create a figure for final notification distances
    plt.figure(figsize=(10, 6))
    if final_notification_distances['bottom']:
        sns.histplot(final_notification_distances['bottom'], kde=True)
        plt.axvline(stats['bottom']['mean'], color='r', linestyle='--', 
                   label=f'Mean: {stats["bottom"]["mean"]:.2f}')
        plt.title(f'Bottom Danger Zone Distance at Final Notification')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.xlim(0, 0.2)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No notifications', ha='center', va='center')
        plt.title(f'Bottom Danger Zone Distance at Final Notification')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_notification_distance_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary_data = []
    for metric, value in stats['bottom'].items():
        summary_data.append({
            'Direction': 'bottom',
            'Metric': metric,
            'Value': value
        })
    
    # Add notification rate
    summary_data.append({
        'Direction': 'All',
        'Metric': 'Notification Rate Before Entry',
        'Value': notification_rate
    })
    
    summary = pd.DataFrame(summary_data)
    
    # Save the summary
    summary.to_csv(f"{output_dir}/final_notification_distance_summary_{policy_name}.csv", index=False)
    
    return {
        'stats': stats,
        'notification_rate': notification_rate
    }

def analyze_notification_rate_per_trajectory(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze notification rate per trajectory and compute the average across all trajectories."""
    # Initialize list to store notification rates per trajectory
    notification_rates_per_trajectory = []
    
    # Analyze each trajectory
    for episode_idx, episode in enumerate(data):
        trajectory = episode['trajectory']
        
        # Count notifications in this trajectory
        notification_count = 0
        total_steps = len(trajectory)
        
        for step in trajectory:
            if step['agent_action_type'] == 2:  # Notification
                notification_count += 1
        
        # Calculate notification rate for this trajectory
        notification_rate = notification_count / total_steps if total_steps > 0 else 0
        notification_rates_per_trajectory.append(notification_rate)
    
    # Calculate statistics
    avg_notification_rate = np.mean(notification_rates_per_trajectory)
    std_notification_rate = np.std(notification_rates_per_trajectory)
    min_notification_rate = np.min(notification_rates_per_trajectory)
    max_notification_rate = np.max(notification_rates_per_trajectory)
    
    # Create a figure for notification rate per trajectory distribution
    plt.figure(figsize=(10, 6))
    plt.hist(notification_rates_per_trajectory, bins=10, alpha=0.7)
    plt.axvline(avg_notification_rate, color='r', linestyle='--', label=f'Mean: {avg_notification_rate:.4f}')
    plt.title(f'Notification Rate Per Trajectory - {policy_name}')
    plt.xlabel('Notification Rate')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/notification_rate_per_trajectory_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Metric': ['Average Notification Rate', 'Std Dev Notification Rate', 'Min Notification Rate', 'Max Notification Rate'],
        'Value': [avg_notification_rate, std_notification_rate, min_notification_rate, max_notification_rate]
    })
    
    # Save the summary
    summary.to_csv(f"{output_dir}/notification_rate_per_trajectory_summary_{policy_name}.csv", index=False)
    
    return {
        'avg_notification_rate': avg_notification_rate,
        'std_notification_rate': std_notification_rate,
        'min_notification_rate': min_notification_rate,
        'max_notification_rate': max_notification_rate,
        'notification_rates_per_trajectory': notification_rates_per_trajectory
    }

def create_comparative_visualizations(policy_results, output_dir):
    """Create comparative visualizations across different policies."""
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define metrics by environment type
    lunar_lander_metrics = [
        ('Final Notification Distance', 'final_notification_distance_stats'),
        ('Notification Rate Before Entry', 'final_notification_distance_stats'),
        ('Danger Zone Entry Rate', 'danger_zone_stats'),
        ('Average Steps to Danger Zone Entry', 'danger_zone_stats'),
    ]
    
    highway_metrics = [
        ('Success Rate', 'success_stats'),
        ('Average Steps to Success', 'success_stats'),
    ]
    
    common_metrics = [
        ('Notification Rate Per Trajectory', 'notification_rate_per_trajectory_stats'),
        ('Notification Rate Distribution', 'notification_rate_per_trajectory_stats'),
        ('Notification Length Frequency', 'per_trajectory_notification_stats'),
        ('Notification Length Selection Rate', 'per_trajectory_notification_stats'),
        ('Human-Agent Agreement Rate', 'interaction_stats'),
        ('Human Overwrite Rate', 'interaction_stats')
    ]
    
    # Determine which metrics to use based on the environment type
    all_metrics = []
    
    # Check if any policy has lunar lander metrics
    has_lunar_lander = False
    for result in policy_results.values():
        if 'final_notification_distance_stats' in result and 'stats' in result['final_notification_distance_stats']:
            has_lunar_lander = True
            break
    
    # Check if any policy has highway metrics
    has_highway = False
    for result in policy_results.values():
        if 'success_stats' in result and 'success_rate' in result['success_stats']:
            has_highway = True
            break
    
    # Add appropriate metrics based on environment type
    if has_lunar_lander:
        all_metrics.extend(lunar_lander_metrics)
    if has_highway:
        all_metrics.extend(highway_metrics)
    all_metrics.extend(common_metrics)
    
    # Create a figure for each metric
    for metric_name, metric_data in all_metrics:
        # Check if any policy has this metric
        has_metric = False
        for policy_name, result in policy_results.items():
            if metric_data in result:
                if metric_name == 'Final Notification Distance' and 'stats' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Notification Rate Before Entry' and 'notification_rate' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Notification Rate Per Trajectory' and 'avg_notification_rate' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Notification Rate Distribution' and 'notification_rates' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Notification Length Frequency' and 'notification_length_frequencies' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Notification Length Selection Rate' and 'notification_length_selection_rates' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Success Rate' and 'success_rate' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Average Steps to Success' and 'avg_steps_to_success' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Danger Zone Entry Rate' and 'entry_rate' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Average Steps to Danger Zone Entry' and 'avg_steps_to_entry' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Human-Agent Agreement Rate' and 'agreement_rate' in result[metric_data]:
                    has_metric = True
                    break
                elif metric_name == 'Human Overwrite Rate' and 'overwrite_rate' in result[metric_data]:
                    has_metric = True
                    break
        
        if not has_metric:
            continue
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot data for each policy
        if metric_name == 'Final Notification Distance':
            # Bar plot of mean final notification distance
            means = []
            stds = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'stats' in result[metric_data]:
                    means.append(result[metric_data]['stats']['bottom']['mean'])
                    stds.append(result[metric_data]['stats']['bottom']['std'])
                    policies.append(policy_name)
            
            if means:
                plt.bar(policies, means, yerr=stds, capsize=5)
                plt.ylabel('Mean Final Notification Distance')
                plt.title('Mean Final Notification Distance by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Notification Rate Before Entry':
            # Bar plot of notification rate before entry
            rates = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'notification_rate' in result[metric_data]:
                    rates.append(result[metric_data]['notification_rate'])
                    policies.append(policy_name)
            
            if rates:
                plt.bar(policies, rates)
                plt.ylabel('Notification Rate Before Entry')
                plt.title('Notification Rate Before Entry by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Notification Rate Per Trajectory':
            # Bar plot of average notification rate per trajectory
            rates = []
            stds = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'avg_notification_rate' in result[metric_data]:
                    rates.append(result[metric_data]['avg_notification_rate'])
                    stds.append(result[metric_data]['std_notification_rate'])
                    policies.append(policy_name)
            
            if rates:
                plt.bar(policies, rates, yerr=stds, capsize=5)
                plt.ylabel('Average Notification Rate Per Trajectory')
                plt.title('Average Notification Rate Per Trajectory by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Notification Rate Distribution':
            # Box plot of notification rate distribution
            data = []
            labels = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'notification_rates' in result[metric_data]:
                    data.append(result[metric_data]['notification_rates'])
                    labels.append(policy_name)
            
            if data:
                plt.boxplot(data, labels=labels)
                plt.ylabel('Notification Rate')
                plt.title('Notification Rate Distribution by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace("", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Notification Length Frequency':
            # Bar plot of notification length frequencies
            lengths = set()
            for result in policy_results.values():
                if metric_data in result and 'notification_length_frequencies' in result[metric_data]:
                    lengths.update(result[metric_data]['notification_length_frequencies'].keys())
            
            if lengths:
                x = np.arange(len(lengths))
                width = 0.8 / len(policy_results)
                
                for i, (policy_name, result) in enumerate(policy_results.items()):
                    if metric_data in result and 'notification_length_frequencies' in result[metric_data]:
                        frequencies = [result[metric_data]['notification_length_frequencies'].get(length, 0) for length in lengths]
                        plt.bar(x + i * width, frequencies, width, label=policy_name)
                
                plt.xlabel('Notification Length')
                plt.ylabel('Frequency')
                plt.title('Notification Length Frequency by Policy')
                plt.xticks(x + width * (len(policy_results) - 1) / 2, lengths)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Notification Length Selection Rate':
            # Bar plot of notification length selection rates
            lengths = set()
            for result in policy_results.values():
                if metric_data in result and 'notification_length_selection_rates' in result[metric_data]:
                    lengths.update(result[metric_data]['notification_length_selection_rates'].keys())
            
            if lengths:
                x = np.arange(len(lengths))
                width = 0.8 / len(policy_results)
                
                for i, (policy_name, result) in enumerate(policy_results.items()):
                    if metric_data in result and 'notification_length_selection_rates' in result[metric_data]:
                        rates = [result[metric_data]['notification_length_selection_rates'].get(length, 0) for length in lengths]
                        plt.bar(x + i * width, rates, width, label=policy_name)
                
                plt.xlabel('Notification Length')
                plt.ylabel('Selection Rate')
                plt.title('Notification Length Selection Rate by Policy')
                plt.xticks(x + width * (len(policy_results) - 1) / 2, lengths)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Success Rate':
            # Bar plot of success rate
            rates = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'success_rate' in result[metric_data]:
                    rates.append(result[metric_data]['success_rate'])
                    policies.append(policy_name)
            
            if rates:
                plt.bar(policies, rates)
                plt.ylabel('Success Rate')
                plt.title('Success Rate by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Average Steps to Success':
            # Bar plot of average steps to success
            steps = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'avg_steps_to_success' in result[metric_data]:
                    steps.append(result[metric_data]['avg_steps_to_success'])
                    policies.append(policy_name)
            
            if steps:
                plt.bar(policies, steps)
                plt.ylabel('Average Steps to Success')
                plt.title('Average Steps to Success by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Danger Zone Entry Rate':
            # Bar plot of danger zone entry rate
            rates = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'entry_rate' in result[metric_data]:
                    rates.append(result[metric_data]['entry_rate'])
                    policies.append(policy_name)
            
            if rates:
                plt.bar(policies, rates)
                plt.ylabel('Danger Zone Entry Rate')
                plt.title('Danger Zone Entry Rate by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Average Steps to Danger Zone Entry':
            # Bar plot of average steps to danger zone entry
            steps = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'avg_steps_to_entry' in result[metric_data]:
                    steps.append(result[metric_data]['avg_steps_to_entry'])
                    policies.append(policy_name)
            
            if steps:
                plt.bar(policies, steps)
                plt.ylabel('Average Steps to Danger Zone Entry')
                plt.title('Average Steps to Danger Zone Entry by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Human-Agent Agreement Rate':
            # Bar plot of human-agent agreement rate
            rates = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'agreement_rate' in result[metric_data]:
                    rates.append(result[metric_data]['agreement_rate'])
                    policies.append(policy_name)
            
            if rates:
                plt.bar(policies, rates)
                plt.ylabel('Human-Agent Agreement Rate')
                plt.title('Human-Agent Agreement Rate by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()
        
        elif metric_name == 'Human Overwrite Rate':
            # Bar plot of human overwrite rate
            rates = []
            policies = []
            
            for policy_name, result in policy_results.items():
                if metric_data in result and 'overwrite_rate' in result[metric_data]:
                    rates.append(result[metric_data]['overwrite_rate'])
                    policies.append(policy_name)
            
            if rates:
                plt.bar(policies, rates)
                plt.ylabel('Human Overwrite Rate')
                plt.title('Human Overwrite Rate by Policy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
                plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory data from collected episodes')
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True, help='Directories containing trajectory data')
    parser.add_argument('--policy_names', type=str, nargs='+', required=True, help='Names for each policy')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save analysis results')
    parser.add_argument('--env_type', type=str, choices=['DangerZoneLunarLander', 'multi-merge-v0', 'auto'], default='auto', 
                        help='Environment type for analysis. Use "auto" to automatically detect from data.')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check that we have the same number of data directories and policy names
    if len(args.data_dirs) != len(args.policy_names):
        raise ValueError("Number of data directories must match number of policy names")

    # Store results for each policy
    policy_results = {}

    for data_dir, policy_name in zip(args.data_dirs, args.policy_names):
        policy_output_dir = os.path.join(args.output_dir, policy_name)
        os.makedirs(policy_output_dir, exist_ok=True)

        # Load data
        data_file = os.path.join(data_dir, 'all_episodes.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Determine environment type if auto
        env_type = args.env_type
        if env_type == 'auto' and data and 'trajectory' in data[0] and data[0]['trajectory']:
            first_step = data[0]['trajectory'][0]
            if 'distance_to_danger' in first_step:
                env_type = 'DangerZoneLunarLander'
            elif 'info' in first_step and 'vehicle_info' in first_step['info']:
                env_type = 'multi-merge-v0'
        
        print(f"Analyzing data for {policy_name} (Environment: {env_type})")

        # Run common analyses
        reward_stats = analyze_rewards(data, policy_output_dir, policy_name)
        length_stats = analyze_episode_lengths(data, policy_output_dir, policy_name)
        action_stats = analyze_action_distribution(data, policy_output_dir, policy_name)
        notification_stats = analyze_notifications(data, policy_output_dir, policy_name)
        interaction_stats = analyze_human_agent_interaction(data, policy_output_dir, policy_name)
        per_trajectory_notification_stats = analyze_per_trajectory_notifications(data, policy_output_dir, policy_name)
        notification_rate_per_trajectory_stats = analyze_notification_rate_per_trajectory(data, policy_output_dir, policy_name)
        
        # Run environment-specific analyses
        success_stats = analyze_success_rate(data, policy_output_dir, policy_name)
        
        # Initialize empty dictionaries for environment-specific stats
        danger_zone_stats = {}
        final_notification_distance_stats = {}
        
        # Only run danger zone analysis if the environment has danger zones
        if env_type == 'DangerZoneLunarLander':
            danger_zone_stats = analyze_danger_zone_interactions(data, policy_output_dir, policy_name)
            final_notification_distance_stats = analyze_final_notification_distance(data, policy_output_dir, policy_name)
        
        # Run trajectory pattern analysis
        trajectory_pattern_stats = analyze_trajectory_patterns(data, policy_output_dir, policy_name)

        # Store results
        policy_results[policy_name] = {
            'env_type': env_type,
            'reward_stats': reward_stats,
            'length_stats': length_stats,
            'action_stats': action_stats,
            'notification_stats': notification_stats,
            'interaction_stats': interaction_stats,
            'per_trajectory_notification_stats': per_trajectory_notification_stats,
            'success_stats': success_stats,
            'danger_zone_stats': danger_zone_stats,
            'notification_rate_per_trajectory_stats': notification_rate_per_trajectory_stats
        }
        
        # Only include final_notification_distance_stats for DangerZoneLunarLander
        if env_type == 'DangerZoneLunarLander':
            policy_results[policy_name]['final_notification_distance_stats'] = final_notification_distance_stats

    # Create comparative analysis
    comparative_data = []
    for policy_name, result in policy_results.items():
        # Get notification length frequencies and selection rates
        length_frequencies = result['per_trajectory_notification_stats'].get('notification_length_frequencies', {})
        length_selection_rates = result['per_trajectory_notification_stats'].get('notification_length_selection_rates', {})
        
        # Create base dictionary with common metrics
        policy_data = {
            'Policy': policy_name,
            'Environment': result['env_type'],
            'Success Rate': result['success_stats'].get('success_rate', 0),
            'Avg Steps to Success': result['success_stats'].get('avg_steps_to_success', 0),
            'Total Notifications': result['notification_stats'].get('total_notifications', 0),
            'Avg Notification Length': result['notification_stats'].get('avg_notification_length', 0),
            'Human Overwrite Rate': result['interaction_stats'].get('overwrite_rate', 0),
            'Avg Notifications Per Trajectory': result['per_trajectory_notification_stats'].get('avg_notifications_per_trajectory', 0),
            'Avg Notification Length Per Trajectory': result['per_trajectory_notification_stats'].get('avg_notification_length_per_trajectory', 0),
            'Avg Overwrite Rate Per Trajectory': result['per_trajectory_notification_stats'].get('avg_overwrite_rate_per_trajectory', 0),
            'Avg Notification Rate Per Trajectory': result['notification_rate_per_trajectory_stats'].get('avg_notification_rate', 0),
            'Std Dev Notification Rate Per Trajectory': result['notification_rate_per_trajectory_stats'].get('std_notification_rate', 0)
        }
        
        # Add danger zone metrics if available
        if result['env_type'] == 'DangerZoneLunarLander' and 'final_notification_distance_stats' in result:
            policy_data.update({
                'Danger Zone Entry Rate': result['danger_zone_stats'].get('entry_rate', 0),
                'Avg Steps to Danger Zone': result['danger_zone_stats'].get('avg_steps_to_entry', 0),
                'Final Notification Distance Mean': result['final_notification_distance_stats'].get('stats', {}).get('bottom', {}).get('mean', 0),
                'Final Notification Distance Std Dev': result['final_notification_distance_stats'].get('stats', {}).get('bottom', {}).get('std', 0),
                'Notification Rate Before Entry': result['final_notification_distance_stats'].get('notification_rate', 0)
            })
        
        # Add notification length frequencies
        for length, freq in length_frequencies.items():
            policy_data[f'Avg Frequency of Length {length}'] = freq
        
        # Add notification length selection rates
        for length, rate in length_selection_rates.items():
            policy_data[f'Selection Rate of Length {length}'] = rate
        
        comparative_data.append(policy_data)

    # Create comparative DataFrame
    comparative_df = pd.DataFrame(comparative_data)
    comparative_df.to_csv(os.path.join(args.output_dir, 'comparative_analysis.csv'), index=False)

    # Create comparative visualizations
    create_comparative_visualizations(policy_results, args.output_dir)

if __name__ == '__main__':
    main() 