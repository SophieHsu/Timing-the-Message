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
    """Analyze interactions with danger zones"""
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
                danger_zone_distances[direction].append(step['distance_to_danger'][direction])
            
            # Check if there's a notification and record the distances at that time
            if step['agent_action_type'] == 2:  # Notification
                for direction in danger_zone_notifications:
                    danger_zone_notifications[direction].append(step['distance_to_danger'][direction])
            
            # Check if entry into any danger zone occurred
            if not entry_occurred:
                for direction, distance in step['distance_to_danger'].items():
                    if distance <= 0:  # Entry into danger zone
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
    """Analyze patterns in the trajectories"""
    # Extract trajectory data
    trajectories = []
    for episode in data:
        trajectory = []
        for step in episode['trajectory']:
            trajectory.append({
                'x': step['observation'][0],
                'y': step['observation'][1],
                'vx': step['observation'][2],
                'vy': step['observation'][3],
                'angle': step['observation'][4],
                'angular_velocity': step['observation'][5],
                'left_leg_contact': step['observation'][6],
                'right_leg_contact': step['observation'][7],
                'notification': step['agent_action_type'] == 2,
                'overwritten': step['overwritten'] == 1
            })
        trajectories.append(trajectory)
    
    # Calculate average trajectory
    max_length = max(len(traj) for traj in trajectories)
    padded_trajectories = [traj + [traj[-1]] * (max_length - len(traj)) for traj in trajectories]
    
    # Fix: Extract x, y, etc. values from each trajectory step
    x_values = [[step['x'] for step in traj] for traj in padded_trajectories]
    y_values = [[step['y'] for step in traj] for traj in padded_trajectories]
    vx_values = [[step['vx'] for step in traj] for traj in padded_trajectories]
    vy_values = [[step['vy'] for step in traj] for traj in padded_trajectories]
    angle_values = [[step['angle'] for step in traj] for traj in padded_trajectories]
    angular_velocity_values = [[step['angular_velocity'] for step in traj] for traj in padded_trajectories]
    
    avg_trajectory = {
        'x': np.mean(x_values, axis=0),
        'y': np.mean(y_values, axis=0),
        'vx': np.mean(vx_values, axis=0),
        'vy': np.mean(vy_values, axis=0),
        'angle': np.mean(angle_values, axis=0),
        'angular_velocity': np.mean(angular_velocity_values, axis=0)
    }
    
    # Plot average trajectory
    plt.figure(figsize=(10, 10))
    plt.plot(avg_trajectory['x'], avg_trajectory['y'], 'b-', label='Average Trajectory')
    
    # Add landing pad
    plt.plot([-0.2, 0.2], [0, 0], 'g-', linewidth=3, label='Landing Pad')
    
    # Add danger zones if available
    if 'danger_zones' in data[0]:
        danger_zones = data[0]['danger_zones']
        for zone in danger_zones:
            x_min, x_max = zone[0]
            y_min, y_max = zone[1]
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                             fill=True, color='red', alpha=0.3))
    
    plt.title(f'Average Trajectory - {policy_name}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.savefig(f"{output_dir}/average_trajectory_{policy_name}.png")
    plt.close()
    
    # Plot velocity components
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(avg_trajectory['vx'], 'b-', label='VX')
    plt.title(f'Average Velocity Components - {policy_name}')
    plt.xlabel('Step')
    plt.ylabel('Velocity X')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(avg_trajectory['vy'], 'r-', label='VY')
    plt.xlabel('Step')
    plt.ylabel('Velocity Y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_velocity_{policy_name}.png")
    plt.close()
    
    # Plot angle and angular velocity
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(avg_trajectory['angle'], 'g-', label='Angle')
    plt.title(f'Average Angle and Angular Velocity - {policy_name}')
    plt.xlabel('Step')
    plt.ylabel('Angle (radians)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(avg_trajectory['angular_velocity'], 'm-', label='Angular Velocity')
    plt.xlabel('Step')
    plt.ylabel('Angular Velocity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_angle_{policy_name}.png")
    plt.close()
    
    return avg_trajectory

def analyze_success_rate(data: List[Dict[str, Any]], output_dir: str, policy_name: str = ""):
    """Analyze success rate and landing statistics"""
    success_count = 0
    crash_count = 0
    timeout_count = 0
    steps_to_success = []
    
    for episode in data:
        last_step = episode['trajectory'][-1]
        if last_step['terminated'] and not last_step['truncated']:
            if last_step['info'].get('success', False):
                success_count += 1
                steps_to_success.append(len(episode['trajectory']))
            else:
                crash_count += 1
        else:
            timeout_count += 1
    
    total_episodes = len(data)
    success_rate = success_count / total_episodes
    crash_rate = crash_count / total_episodes
    timeout_rate = timeout_count / total_episodes
    
    # Calculate average steps to success
    avg_steps_to_success = np.mean(steps_to_success) if steps_to_success else 0
    
    # Create a pie chart
    plt.figure(figsize=(10, 8))
    plt.pie([success_count, crash_count, timeout_count], 
            labels=['Success', 'Crash', 'Timeout'],
            autopct='%1.1f%%',
            colors=['green', 'red', 'orange'])
    plt.title(f'Episode Outcomes - {policy_name}')
    plt.savefig(f"{output_dir}/outcome_distribution_{policy_name}.png")
    plt.close()
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Outcome': ['Success', 'Crash', 'Timeout'],
        'Count': [success_count, crash_count, timeout_count],
        'Rate': [success_rate, crash_rate, timeout_rate]
    })
    
    # Save the summary
    summary.to_csv(f"{output_dir}/outcome_summary_{policy_name}.csv", index=False)
    
    return {
        'success_count': success_count,
        'crash_count': crash_count,
        'timeout_count': timeout_count,
        'success_rate': success_rate,
        'crash_rate': crash_rate,
        'timeout_rate': timeout_rate,
        'avg_steps_to_success': avg_steps_to_success
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
    
    # Create a confusion matrix for human vs agent actions
    confusion_matrix = np.zeros((4, 4))
    for h, a in zip(human_actions, agent_actions):
        confusion_matrix[h, a] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=['No-op', 'Left', 'Main', 'Right'],
                yticklabels=['No-op', 'Left', 'Main', 'Right'])
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
    
    # Analyze each trajectory
    for episode_idx, episode in enumerate(data):
        trajectory = episode['trajectory']
        
        # Count notifications in this trajectory
        notification_steps = []
        notification_lengths = []
        overwritten_after_notification = []
        
        for i, step in enumerate(trajectory):
            if step['agent_action_type'] == 2:  # Notification
                notification_steps.append(i)
                notification_lengths.append(step['agent_action_length'])
                
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
        
        # Store statistics
        notifications_per_trajectory.append(num_notifications)
        notification_lengths_per_trajectory.append(avg_notification_length)
        overwrite_rates_per_trajectory.append(overwrite_rate)
    
    # Calculate overall statistics
    avg_notifications_per_trajectory = np.mean(notifications_per_trajectory)
    std_notifications_per_trajectory = np.std(notifications_per_trajectory)
    avg_notification_length_per_trajectory = np.mean(notification_lengths_per_trajectory)
    std_notification_length_per_trajectory = np.std(notification_lengths_per_trajectory)
    avg_overwrite_rate_per_trajectory = np.mean(overwrite_rates_per_trajectory)
    std_overwrite_rate_per_trajectory = np.std(overwrite_rates_per_trajectory)
    
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
    
    # Save the summary
    summary.to_csv(f"{output_dir}/per_trajectory_notification_summary_{policy_name}.csv", index=False)
    
    return {
        'avg_notifications_per_trajectory': avg_notifications_per_trajectory,
        'std_notifications_per_trajectory': std_notifications_per_trajectory,
        'avg_notification_length_per_trajectory': avg_notification_length_per_trajectory,
        'std_notification_length_per_trajectory': std_notification_length_per_trajectory,
        'avg_overwrite_rate_per_trajectory': avg_overwrite_rate_per_trajectory,
        'std_overwrite_rate_per_trajectory': std_overwrite_rate_per_trajectory
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
    """Create comparative visualizations for all policies."""
    # Create a figure for final notification distance comparison
    plt.figure(figsize=(10, 6))
    
    # Plot mean final notification distance for bottom danger zone
    for policy_name, result in policy_results.items():
        plt.bar(policy_name, result['final_notification_distance_stats']['stats']['bottom']['mean'],
               yerr=result['final_notification_distance_stats']['stats']['bottom']['std'], capsize=10)
    plt.title('Mean Final Notification Distance to Bottom Danger Zone')
    plt.ylabel('Mean Distance')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_notification_distance_comparison.png'))
    plt.close()
    
    # Create a figure for notification rate before entry comparison
    plt.figure(figsize=(10, 6))
    for policy_name, result in policy_results.items():
        plt.bar(policy_name, result['final_notification_distance_stats']['notification_rate'])
    plt.title('Notification Rate Before Bottom Danger Zone Entry')
    plt.ylabel('Notification Rate')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'notification_rate_before_entry_comparison.png'))
    plt.close()
    
    # Create a figure for notification rate per trajectory comparison
    plt.figure(figsize=(10, 6))
    for policy_name, result in policy_results.items():
        plt.bar(policy_name, result['notification_rate_per_trajectory_stats']['avg_notification_rate'],
               yerr=result['notification_rate_per_trajectory_stats']['std_notification_rate'], capsize=10)
    plt.title('Average Notification Rate Per Trajectory')
    plt.ylabel('Average Notification Rate')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'notification_rate_per_trajectory_comparison.png'))
    plt.close()
    
    # Create a figure for notification rate distribution comparison
    plt.figure(figsize=(12, 8))
    for policy_name, result in policy_results.items():
        plt.hist(result['notification_rate_per_trajectory_stats']['notification_rates_per_trajectory'], 
                bins=10, alpha=0.5, label=policy_name)
    plt.title('Notification Rate Distribution Per Trajectory')
    plt.xlabel('Notification Rate')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'notification_rate_distribution_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory data from collected episodes')
    parser.add_argument('--data_dirs', type=str, nargs='+', required=True, help='Directories containing trajectory data')
    parser.add_argument('--policy_names', type=str, nargs='+', required=True, help='Names for each policy')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save analysis results')
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

        # Run analyses
        success_stats = analyze_success_rate(data, policy_output_dir, policy_name)
        danger_zone_stats = analyze_danger_zone_interactions(data, policy_output_dir, policy_name)
        notification_stats = analyze_notifications(data, policy_output_dir, policy_name)
        interaction_stats = analyze_human_agent_interaction(data, policy_output_dir, policy_name)
        per_trajectory_notification_stats = analyze_per_trajectory_notifications(data, policy_output_dir, policy_name)
        final_notification_distance_stats = analyze_final_notification_distance(data, policy_output_dir, policy_name)
        notification_rate_per_trajectory_stats = analyze_notification_rate_per_trajectory(data, policy_output_dir, policy_name)

        # Store results
        policy_results[policy_name] = {
            'success_stats': success_stats,
            'danger_zone_stats': danger_zone_stats,
            'notification_stats': notification_stats,
            'interaction_stats': interaction_stats,
            'per_trajectory_notification_stats': per_trajectory_notification_stats,
            'final_notification_distance_stats': final_notification_distance_stats,
            'notification_rate_per_trajectory_stats': notification_rate_per_trajectory_stats
        }

    # Create comparative analysis
    comparative_data = []
    for policy_name, result in policy_results.items():
        comparative_data.append({
            'Policy': policy_name,
            'Success Rate': result['success_stats']['success_rate'],
            'Avg Steps to Success': result['success_stats']['avg_steps_to_success'],
            'Danger Zone Entry Rate': result['danger_zone_stats']['entry_rate'],
            'Avg Steps to Danger Zone': result['danger_zone_stats']['avg_steps_to_entry'],
            'Total Notifications': result['notification_stats']['total_notifications'],
            'Avg Notification Length': result['notification_stats']['avg_notification_length'],
            'Human Overwrite Rate': result['interaction_stats']['overwrite_rate'],
            'Avg Notifications Per Trajectory': result['per_trajectory_notification_stats']['avg_notifications_per_trajectory'],
            'Avg Notification Length Per Trajectory': result['per_trajectory_notification_stats']['avg_notification_length_per_trajectory'],
            'Avg Overwrite Rate Per Trajectory': result['per_trajectory_notification_stats']['avg_overwrite_rate_per_trajectory'],
            'Final Notification Distance Mean': result['final_notification_distance_stats']['stats']['bottom']['mean'],
            'Final Notification Distance Std Dev': result['final_notification_distance_stats']['stats']['bottom']['std'],
            'Notification Rate Before Entry': result['final_notification_distance_stats']['notification_rate'],
            'Avg Notification Rate Per Trajectory': result['notification_rate_per_trajectory_stats']['avg_notification_rate'],
            'Std Dev Notification Rate Per Trajectory': result['notification_rate_per_trajectory_stats']['std_notification_rate']
        })

    # Create comparative DataFrame
    comparative_df = pd.DataFrame(comparative_data)
    comparative_df.to_csv(os.path.join(args.output_dir, 'comparative_analysis.csv'), index=False)

    # Create comparative visualizations
    create_comparative_visualizations(policy_results, args.output_dir)

if __name__ == '__main__':
    main() 