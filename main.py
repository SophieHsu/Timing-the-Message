import argparse
import numpy as np
import gymnasium_envs
import gymnasium as gym


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="highway_fast")
    args = parser.parse_args()
    
    # Initialize the environment
    if args.env == "lunarlander":
        env = gym.make("gymnasium_envs/NotiLunarLander", render_mode="rgb_array")
    elif args.env == "highway":
        env = gym.make("gymnasium_envs/NotiHighway", render_mode="rgb_array")
    elif args.env == "highway_fast":
        env = gym.make("gymnasium_envs/NotiHighwayFast", render_mode="rgb_array")
    else:
        raise ValueError(f"Environment {args.env} not found")
    observation, info = env.reset()

    # Initialize the agent
    noti_agent = NotiAgent(env)
    human_agent = HumanAgent(env)

    # Run the environment
    done = False
    while not done:
        action = noti_agent.action(observation)
        human_action = human_agent.action(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)

        env.render()

    env.close()