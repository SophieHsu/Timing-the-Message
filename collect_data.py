import os
import torch
import tyro
import json
import time
import wandb
import numpy as np
from pathlib import Path
from gymnasium.vector import SyncVectorEnv
import gymnasium as gym
import gymnasium_envs
import highway_env
from tqdm import tqdm

from src.configs.args import Args
from src.utils.util import make_env
from src.agents.mlp import NotifierMLPAgent
from src.agents.lstm import NotifierLSTMAgent
from src.agents.transformers import TransformerAgent
from src.agents.humans import HumanAgent, HumanDriverAgent

os.environ["OFFSCREEN_RENDERING"] = "1"

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    # Parse command line arguments
    args = tyro.cli(Args)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.device is None:
        args.device = device
    
    # Number of parallel environments to run
    num_envs = args.num_envs
    print(f"Running {num_envs} environments in parallel")
    
    # Create vectorized environment
    envs = SyncVectorEnv([make_env(args.env_id, i, False, args.exp_name) for i in range(num_envs)])
    
    # Create agent based on args
    if args.agent_type == "mlp":
        agent = NotifierMLPAgent(envs, args).to(device)
    elif args.agent_type == "lstm":
        agent = NotifierLSTMAgent(envs, args).to(device)
    elif args.agent_type == "transformer":
        agent = TransformerAgent(envs, args).to(device)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")
    
    # Load the trained agent
    api = wandb.Api()
    run = api.run(f"yachuanh/timing/{args.model_run_id}")

    # Create output directory for data
    output_dir = Path(f"data/{args.model_run_id}_convey{run.config['human_comprehend_bool']}_delay{run.config['human_reaction_delay']}_gtconvey{args.human_comprehend_bool}_gtdelay{args.human_reaction_delay}_{int(time.time())}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = run.config['filepath'] + "/agent.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()
    
    # Set up for data collection
    single_observation_space = (np.array(envs.single_observation_space.shape).prod() + (envs.single_action_space.shape[0]-1))*args.human_utterance_memory_length
    next_agent_obs = torch.zeros((600, num_envs) + (single_observation_space,)).to(device)
    full_next_agent_obs = torch.zeros((600*10, num_envs) + (single_observation_space,)).to(device)
    
    # Create a single human agent for all environments
    if args.human_agent_type == "IDM":
        human_agent = HumanDriverAgent(envs, args, device)
    else:
        human_agent = HumanAgent(envs, args, device)
    
    # Initialize data collection
    total_episodes = 100
    all_episodes_data = []
    episode_count = 0
    
    # Initialize environment state
    obs, infos = envs.reset()
    total_rewards = np.zeros(num_envs)
    steps = np.zeros(num_envs, dtype=int)
    episode_data = [[] for _ in range(num_envs)]
    dones = np.zeros(num_envs, dtype=bool)
    
    # Run episodes and collect data
    with tqdm(total=total_episodes, desc="Collecting episodes") as pbar:
        while episode_count < total_episodes:
            # Get agent actions for all environments
            reshape_next_obs = obs.reshape(num_envs, -1)
            curr_agent_obs = torch.cat([torch.Tensor(reshape_next_obs).to(args.device), torch.Tensor(infos['utterance']).to(args.device)], dim=1)
            prev_agent_obs = full_next_agent_obs[-1].reshape(num_envs, args.human_utterance_memory_length, -1)[:,1:]
            next_agent_obs = torch.cat([prev_agent_obs, curr_agent_obs.unsqueeze(1)], dim=1).reshape(num_envs, -1)
            agent_actions, _, _, _ = agent.get_action_and_value(next_agent_obs)
            
            # Get human actions for all environments
            human_actions, overwrite_flags = human_agent.get_action(torch.Tensor(obs).to(device), infos["utterance"])
            
            # Concatenate actions for all environments
            actions = np.concatenate([
                agent_actions.cpu().numpy(), 
                human_actions.reshape(-1, 1), 
                overwrite_flags.reshape(-1, 1)
            ], axis=1)
            
            # Execute actions for all environments
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            next_dones = np.logical_or(terminations, truncations)
            
            # Process each environment
            for env_idx in range(num_envs):
                if not dones[env_idx]:
                    # Update rewards and steps
                    total_rewards[env_idx] += rewards[env_idx]
                    steps[env_idx] += 1
                    
                    # Collect detailed trajectory data
                    trajectory_step = {
                        'step': int(steps[env_idx]),
                        'agent_action_type': infos["utterance"][env_idx][0],
                        'agent_action': infos["utterance"][env_idx][1],
                        'agent_action_length': infos["utterance"][env_idx][2],
                        'human_action': human_actions[env_idx].item(),
                        'overwritten': overwrite_flags[env_idx],
                        'reward': rewards[env_idx],
                        'observation': obs[env_idx].tolist(),
                        'distance_to_danger': {
                            'left': obs[env_idx][8], 
                            'right': obs[env_idx][9], 
                            'top': obs[env_idx][10], 
                            'bottom': obs[env_idx][11]
                        },
                        'next_observation': next_obs[env_idx].tolist(),
                        'next_distance_to_danger': {
                            'left': next_obs[env_idx][8], 
                            'right': next_obs[env_idx][9], 
                            'top': next_obs[env_idx][10], 
                            'bottom': next_obs[env_idx][11]
                        },
                        'terminated': bool(terminations[env_idx]),
                        'truncated': bool(truncations[env_idx]),
                        'info': {k: v[env_idx].tolist() if isinstance(v, np.ndarray) else v for k, v in infos.items()}
                    }
                    
                    # Add to episode data
                    episode_data[env_idx].append(trajectory_step)
                    
                    # Check if environment is done
                    if next_dones[env_idx]:
                        dones[env_idx] = True
                        
                        # Add episode summary
                        episode_summary = {
                            'episode_idx': episode_count,
                            'env_idx': env_idx,
                            'total_reward': float(total_rewards[env_idx]),
                            'num_steps': int(steps[env_idx]),
                            'trajectory': episode_data[env_idx]
                        }
                        
                        # Add to all episodes data
                        all_episodes_data.append(episode_summary)
                        episode_count += 1
                        pbar.update(1)
                        
                        # Reset human agent for this environment
                        o, i = envs.envs[env_idx].reset()
                        next_obs[env_idx] = o
                        for k in i.keys():
                            infos[k][env_idx] = i[k] 
                        human_agent.reset()
            
            # Update observations
            obs = next_obs
            
            # If all environments are done, reset them
            if np.all(dones):
                # Save data periodically
                if episode_count % 10 == 0 or episode_count >= total_episodes:
                    with open(output_dir / f"episodes_{episode_count}.json", 'w') as f:
                        json.dump(all_episodes_data, f, indent=2, cls=NumpyEncoder)
                    print(f"Saved data for {episode_count} episodes")
                
                # Reset environments
                obs, infos = envs.reset()
                total_rewards = np.zeros(num_envs)
                steps = np.zeros(num_envs, dtype=int)
                episode_data = [[] for _ in range(num_envs)]
                dones = np.zeros(num_envs, dtype=bool)
    
    # Save final data
    with open(output_dir / "all_episodes.json", 'w') as f:
        json.dump(all_episodes_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"Data collection complete. Collected {episode_count} episodes.")
    print(f"Data saved to {output_dir}")
    
    # Print summary statistics
    total_rewards = [ep['total_reward'] for ep in all_episodes_data]
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    
    # Close environment
    envs.close()

if __name__ == "__main__":
    main() 