import gymnasium as gym
import numpy as np
from highway_env.envs.merge_env import HeuristicNotiMultiMergeEnv

def main():
    # Configure the environment
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 8,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-10, 750],
                "y": [-20, 20],
                "vx": [-5, 40],
                "vy": [-5, 40]
            },
            "absolute": False,
            "order": "sorted"
        },
        "action": {
            "type": "NotiAction",
            "discretization": "simple"
        },
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": True,
        "render_agent": True,
        "offscreen_rendering": False,
        "vehicle_class": "HeuristicNotiIDMVehicle",
        "target_speeds": [5, 10, 15, 20, 25, 30, 35, 40],
        "human_utterance_memory_length": 10,
        "merge_vehicles_type": "highway_env.vehicle.behavior.MergeIDMVehicle",
        "max_episode_steps": 1000,
        "reward_speed_range": [5, 40],
        "high_speed_reward": 1.0,
        "collision_reward": -2.0,
        "progress_reward": 0.1,
        "completion_bonus": 5.0,
        "merging_speed_reward": -0.3,
        "lane_change_reward": -0.05,
        "right_lane_reward": 0.2,
        "noti_penalty": -0.3
    }
    
    # Set default configuration
    HeuristicNotiMultiMergeEnv.default_config = lambda: config
    
    # Create the environment
    env = HeuristicNotiMultiMergeEnv()
    
    # Reset the environment
    obs, info = env.reset()
    
    # Run for a few episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not (done or truncated):
            # The environment uses a heuristic approach for notifications
            # We just need to provide the human action
            # -1 means no human action (let the heuristic agent decide)
            action = -1
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Print information
            print(f"Step {step}")
            print(f"Reward: {reward:.2f}")
            if 'utterance' in info:
                print(f"Notification: {info['utterance']}")
            if 'action' in info:
                print(f"Action: {info['action']}")
            print("-" * 50)
            
            total_reward += reward
            step += 1
            
            # Render the environment
            env.render()
        
        print(f"Episode finished after {step} steps")
        print(f"Total reward: {total_reward:.2f}")
        
        # Reset for next episode
        obs, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    main() 