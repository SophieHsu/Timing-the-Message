import argparse
import gymnasium as gym
import gymnasium_envs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="highway_fast")
    args = parser.parse_args()
    
    if args.env == "lunarlander":
        env = gym.make("gymnasium_envs/NotiLunarLander", render_mode="rgb_array")
    elif args.env == "highway":
        env = gym.make("gymnasium_envs/NotiHighway", render_mode="rgb_array")
    elif args.env == "highway_fast":
        env = gym.make("gymnasium_envs/NotiHighwayFast", render_mode="rgb_array")
    else:
        raise ValueError(f"Environment {args.env} not found")
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        env.render()
    env.close()