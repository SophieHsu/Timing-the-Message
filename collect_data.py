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
from src.agents.heuristic import HeuristicAgent
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

    # Update args from envs
    if args.env_id != "steakhouse":
        args.noti_action_length = envs.envs[0].unwrapped.noti_action_length
    else:
        args.noti_action_length = envs.noti_action_length
    
    # Create agent based on args
    if args.agent_type == "mlp":
        agent = NotifierMLPAgent(args, envs.single_observation_space, envs.single_action_space, args.noti_action_length).to(device)
    elif args.agent_type == "lstm":
        agent = NotifierLSTMAgent(args, envs.single_observation_space, envs.single_action_space).to(device)
    elif args.agent_type == "transformer":
        agent = TransformerAgent(envs, args).to(device)
    elif args.agent_type == "heuristic":
        agent = HeuristicAgent(envs, args)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")
    
    # For heuristic agent, we don't need to load weights from wandb
    if args.agent_type != "heuristic":
        # Load the trained agent
        api = wandb.Api()
        run = api.run(f"{args.wandb_entity}/timing/{args.model_run_id}")
        model_path = run.config['filepath'] + "/agent.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()

        output_dir = Path(f"data/{args.env_id}/{args.model_run_id}_convey{run.config['human_comprehend_bool']}_delay{run.config['human_reaction_delay']}_gtconvey{args.human_comprehend_bool}_gtdelay{args.human_reaction_delay}_{int(time.time())}")

    # Create output directory for data
    if args.agent_type == 'heuristic':
        output_dir = Path(f"data/{args.env_id}/heuristic_gtconvey{args.human_comprehend_bool}_gtdelay{args.human_reaction_delay}_{int(time.time())}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
                        'next_observation': next_obs[env_idx].tolist(),
                        'terminated': bool(terminations[env_idx]),
                        'truncated': bool(truncations[env_idx]),
                        'info': {k: v[env_idx].tolist() if isinstance(v, np.ndarray) else v for k, v in infos.items()},
                    }
                    
                    # Add environment-specific data
                    if "DangerZoneLunarLander" in args.env_id:
                        # Add danger zone distances for Lunar Lander
                        trajectory_step['distance_to_danger'] = {
                            'left': None if len(obs[env_idx]) < 9 else obs[env_idx][8], 
                            'right': None if len(obs[env_idx]) < 10 else obs[env_idx][9], 
                            'top': None if len(obs[env_idx]) < 11 else obs[env_idx][10], 
                            'bottom': None if len(obs[env_idx]) < 12 else obs[env_idx][11]
                        }
                        trajectory_step['next_distance_to_danger'] = {
                            'left': None if len(next_obs[env_idx]) < 9 else next_obs[env_idx][8], 
                            'right': None if len(next_obs[env_idx]) < 10 else next_obs[env_idx][9], 
                            'top': None if len(next_obs[env_idx]) < 11 else next_obs[env_idx][10], 
                            'bottom': None if len(next_obs[env_idx]) < 12 else next_obs[env_idx][11]
                        }
                    elif "multi-merge-v0" in args.env_id:
                        # Add vehicle information for highway environment
                        if 'vehicle_info' in infos:
                            trajectory_step['vehicle_info'] = infos['vehicle_info'][env_idx]
                        
                        # Add road information if available
                        if hasattr(envs.envs[env_idx].unwrapped, 'road'):
                            road = envs.envs[env_idx].unwrapped.road
                            trajectory_step['road_info'] = {
                                'network': str(road.network),
                                'vehicles': [str(v) for v in road.vehicles],
                                'vehicle_count': len(road.vehicles)
                            }
                        
                        # Add vehicle velocity information
                        trajectory_step['vehicle_velocity'] = {
                            'vx': obs[env_idx][0][3],  # vx is at index 3 in the observation
                            'vy': obs[env_idx][0][4],  # vy is at index 4 in the observation
                            'next_vx': next_obs[env_idx][0][3],
                            'next_vy': next_obs[env_idx][0][4]
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
                        
                        # Add environment-specific summary data
                        if args.env_id == "DangerZoneLunarLander":
                            try:
                                episode_summary['danger_zones'] = envs.envs[env_idx].unwrapped.danger_zones
                            except:
                                pass
                        elif args.env_id == "multi-merge-v0":
                            try:
                                # Add road information to summary
                                road = envs.envs[env_idx].unwrapped.road
                                episode_summary['road_summary'] = {
                                    'network': str(road.network),
                                    'vehicle_count': len(road.vehicles),
                                    'crashed': envs.envs[env_idx].unwrapped.vehicle.crashed
                                }
                            except:
                                pass
                        
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