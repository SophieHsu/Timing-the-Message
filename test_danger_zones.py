import gymnasium as gym
import numpy as np
import time
from PIL import Image
from gymnasium.envs.registration import register
from gymnasium_envs.envs.lunar_lander import RandomDangerZoneLunarLander

# Register the environment
register(
    id="RandomDangerZoneLunarLander-v0",
    entry_point="gymnasium_envs.envs.lunar_lander:RandomDangerZoneLunarLander",
    max_episode_steps=1000,
)

def test_danger_zones():
    # Create the environment
    env = gym.make("RandomDangerZoneLunarLander-v0", render_mode="rgb_array")
    unwrapped_env = env.unwrapped
    
    # Force each danger zone configuration and visualize it
    for i in range(len(unwrapped_env.possible_danger_zones)):
        print(f"\nTesting Danger Zone Configuration {i + 1}")
        
        # Set specific danger zone configuration
        unwrapped_env.danger_zones = unwrapped_env.possible_danger_zones[i]
        obs, info = env.reset()
        
        # Render and save the image
        frame = env.render()
        img = Image.fromarray(frame)
        img.save(f"danger_zone_config_{i+1}.png")
        
        # Print the danger zone coordinates for verification
        print("Danger Zone Coordinates:")
        for j, zone in enumerate(unwrapped_env.danger_zones):
            print(f"Zone {j + 1}: X: {zone[0]}, Y: {zone[1]}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    test_danger_zones() 