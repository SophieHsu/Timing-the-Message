import torch
import gymnasium as gym
import numpy as np
from typing import Callable
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.humans import HumanAgent, HumanDriverAgent
from agents.heuristic import HeuristicAgent
import copy
import matplotlib.pyplot as plt
import json
from pathlib import Path

class BaseEvaluator:
    def __init__(self, args, run_name):
        self.args = copy.deepcopy(args)
        self.run_name = run_name
        self.visualize = False  # Default to not visualizing
        self.trajectory_data = []  # Store trajectory data for visualization

    def compute_next_agent_obs(self, next_obs, infos):
        if self.args.agent_obs_mode == "history":
            reshape_next_obs = next_obs.reshape(self.args.num_envs, -1)
            curr_agent_obs = torch.cat([reshape_next_obs, torch.Tensor(infos['utterance']).to(self.args.device)], dim=1)
            prev_agent_obs = self.full_next_agent_obs[-1].reshape(self.args.num_envs, self.args.human_utterance_memory_length, -1)[:,1:]
            next_agent_obs = torch.cat([prev_agent_obs, curr_agent_obs.unsqueeze(1)], dim=1).reshape(self.args.num_envs, -1)
        else:
            reshape_next_obs = next_obs.reshape(self.args.num_envs, -1)
            next_agent_obs = torch.cat([reshape_next_obs, torch.Tensor(infos['utterance']).to(self.args.device)], dim=1)

        return next_agent_obs

    def visualize_trajectory(self, episode_idx, trajectory_data):
        """Visualize the trajectory with different colors for overwritten actions"""
        if not self.visualize:
            return
            
        # Create directory for plots if it doesn't exist
        os.makedirs(f"plots/{self.run_name}", exist_ok=True)
        
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
        ax1.set_title('Notifications and Human Actions Over Time')
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

        no_op_steps = [i for i, t in enumerate(agent_actions_type) if t == 1]
        no_op_actions = [0.5 for i in no_op_steps]
        ax1.scatter(no_op_steps, no_op_actions, color='blue', marker='x', label='Cont.', alpha=0.7)
        
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
        plot_path = f"plots/{self.run_name}/episode_{episode_idx}.png"
        plt.savefig(plot_path)
        
        # # Log to wandb if available
        # try:
        #     import wandb
        #     if wandb.run is not None:
        #         wandb.log({
        #             f"trajectory/episode_{episode_idx}": wandb.Image(plot_path),
        #             f"trajectory/episode_{episode_idx}_total_reward": sum(rewards),
        #             f"trajectory/episode_{episode_idx}_num_overwritten": sum(overwritten)
        #         })
        # except (ImportError, AttributeError):
        #     pass  # wandb not available or not initialized
        
        plt.close()
        
        # Save trajectory data as JSON
        with open(f"plots/{self.run_name}/episode_{episode_idx}.json", 'w') as f:
            # Convert NumPy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
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
            
            json.dump(convert_numpy_types(trajectory_data), f, indent=2)

    def evaluate(self,
        model_path: str,
        make_env: Callable,
        eval_episodes: int,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        capture_video: bool = True,
        visualize: bool = False,
    ):
        self.visualize = visualize
        
        envs = gym.vector.SyncVectorEnv([make_env(self.args.env_id, 0, capture_video, self.run_name)])
        agent = model(envs, self.args).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()

        self.args.num_envs = 1 # Override num_envs to 1 for evaluation
        self.args.device = device
        self.next_agent_obs = torch.zeros((self.args.max_episode_steps, self.args.num_envs) + (agent.single_observation_space,)).to(device)
        self.full_next_agent_obs = torch.zeros((self.args.max_episode_steps*10, self.args.num_envs) + (agent.single_observation_space,)).to(device)

        human_agent = None
        if self.args.human_agent_type is not None and self.args.human_agent_type != "IDM":
            human_agent = HumanAgent(envs, self.args, device)
        elif self.args.human_agent_type is not None and self.args.human_agent_type == "IDM":
            human_agent = HumanDriverAgent(envs, self.args, device)

        obs, infos = envs.reset()
        episodic_returns = []
        # New metrics
        episodic_type2_counts = []
        episodic_overwritten_counts = []
        episodic_action_length_varieties = {}
        
        total_reward = 0
        # Initialize episode-specific metrics
        type2_count = 0
        overwritten_count = 0
        action_length_counts = {}  # Dictionary to track frequency of each action length
        
        step = 0
        episode_idx = 0
        
        while len(episodic_returns) < eval_episodes:
            # Initialize trajectory data for this episode
            if step == 0:
                trajectory_data = []
                # Reset episode-specific metrics
                type2_count = 0
                overwritten_count = 0
                action_length_counts = {}  # Reset action length counts for new episode
            
            # Get agent action
            if human_agent is not None:
                next_agent_obs = self.compute_next_agent_obs(torch.Tensor(obs).to(device), infos)
            else:
                next_agent_obs = torch.Tensor(obs).to(device)

            agent_actions, _, _, _ = agent.get_action_and_value(torch.Tensor(next_agent_obs).to(device))
            self.next_agent_obs[step] = next_agent_obs
            self.full_next_agent_obs[step] = next_agent_obs
            # Get human action if applicable
            human_action = None
            overwrite_flag = False
            
            if human_agent is not None:
                human_actions, overwrite_flag = human_agent.get_action(torch.Tensor(obs).to(device), infos["utterance"])
                actions = np.concatenate([agent_actions, human_actions.reshape(-1,1), overwrite_flag.reshape(-1,1)], axis=1)[0]
                human_action = human_actions.item()
            else:
                actions = (0, 0, 0, agent_actions.cpu().numpy().item(), 0)
                human_action = None
            
            # Execute action
            next_obs, reward, terminations, truncations, info = envs.envs[0].step(actions)
            infos = {k: np.array([info[k]]) for k in info}
            next_done = np.logical_or(terminations, truncations)
            total_reward += reward

            # Track additional metrics
            agent_action_type = info["utterance"][0]
            agent_action_length = info["utterance"][2]
            
            if agent_action_type == 2:
                type2_count += 1
                
            if overwrite_flag:
                overwritten_count += 1
                
            # Track action length frequencies
            if agent_action_length not in action_length_counts:
                action_length_counts[agent_action_length] = 0
            action_length_counts[agent_action_length] += 1

            # Log trajectory data
            trajectory_step = {
                'step': step,
                'agent_action_type': agent_action_type,
                'agent_action': info["utterance"][1],
                'agent_action_length': agent_action_length,
                'human_action': human_action,
                'overwritten': overwrite_flag,
                'reward': reward
            }
            
            # Update trajectory data with reward
            trajectory_data.append(trajectory_step)
            
            next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor([next_done]).to(device)

            if next_done:
                episodic_returns += [total_reward]
                # Add the new metrics for this episode
                episodic_type2_counts += [type2_count]
                episodic_overwritten_counts += [overwritten_count]
                for key, value in action_length_counts.items():
                    if key not in episodic_action_length_varieties:
                        episodic_action_length_varieties[key] = []
                    episodic_action_length_varieties[key] += [value]
                
                # Visualize trajectory if enabled
                if self.visualize:
                    self.visualize_trajectory(episode_idx, trajectory_data)
                
                obs, infos = envs.reset()
                next_obs = torch.Tensor(obs).to(device)
                total_reward = 0
                step = 0
                episode_idx += 1
                if human_agent is not None:
                    human_agent.reset()
            else:
                step += 1
                
            obs = next_obs

        return episodic_returns, episodic_type2_counts, episodic_overwritten_counts, episodic_action_length_varieties


class LSTMEvaluator(BaseEvaluator):
    def __init__(self, args, run_name):
        super().__init__(args, run_name)
    
    def evaluate(self,
        model_path: str,
        make_env: Callable,
        eval_episodes: int,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        capture_video: bool = True,
        visualize: bool = False,
    ):
        self.visualize = visualize
        
        envs = gym.vector.SyncVectorEnv([make_env(self.args.env_id, 0, capture_video, self.run_name)])
        agent = model(envs, self.args).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()

        self.args.num_envs = 1 # Override num_envs to 1 for evaluation
        self.args.device = device
        self.next_agent_obs = torch.zeros((600, self.args.num_envs) + (agent.single_observation_space,)).to(device)
        self.full_next_agent_obs = torch.zeros((600*10, self.args.num_envs) + (agent.single_observation_space,)).to(device)
        human_agent = None
        if self.args.human_agent_type is not None and self.args.human_agent_type != "None":
            human_agent = HumanAgent(envs, self.args, device)

        next_done = torch.zeros(1).to(device)
        total_reward = np.zeros(1)
        init_flag = True

        episodic_returns = []
        # New metrics
        episodic_type2_counts = []
        episodic_overwritten_counts = []
        episodic_action_length_varieties = {}
        
        n_steps = []
        step = 0
        episode_idx = 0
        
        # Initialize episode-specific metrics
        type2_count = 0
        overwritten_count = 0
        action_length_counts = {}  # Dictionary to track frequency of each action length
        
        while len(episodic_returns) < eval_episodes:
            # Initialize trajectory data for this episode
            if next_done[0] or init_flag:
                if next_done[0]:
                    # log before moving on to next episode
                    # print(f"eval_episode={len(episodic_returns)}, episodic_return={total_reward[0]}")
                    episodic_returns += [total_reward[0]]
                    # Add the new metrics for this episode
                    episodic_type2_counts += [type2_count]
                    episodic_overwritten_counts += [overwritten_count]
                    for key, value in action_length_counts.items():
                        if key not in episodic_action_length_varieties:
                            episodic_action_length_varieties[key] = []
                        episodic_action_length_varieties[key] += [value]
                    n_steps.append(n)
                    
                    # Visualize trajectory if enabled
                    if self.visualize and len(trajectory_data) > 0:
                        self.visualize_trajectory(episode_idx, trajectory_data)
                        episode_idx += 1
                
                n = 0
                step = 0
                obs, infos = envs.reset()
                if human_agent is not None:
                    human_agent.reset()
                next_obs = torch.Tensor(obs).to(device)
                next_done = torch.zeros(1).to(device)
                next_lstm_state = (
                    torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
                    torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
                )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
                total_reward = np.zeros(1)
                init_flag = False
                
                # Initialize trajectory data for new episode
                trajectory_data = []
                # Reset episode-specific metrics
                type2_count = 0
                overwritten_count = 0
                action_length_counts = {}  # Reset action length counts for new episode

            if human_agent is not None:
                next_agent_obs = self.compute_next_agent_obs(next_obs, infos)
            else:
                next_agent_obs = next_obs
                
            # Get agent action
            agent_actions, _, _, _, next_lstm_state = agent.get_action_and_value(next_agent_obs, next_lstm_state, next_done)
            self.next_agent_obs[step] = next_agent_obs
            self.full_next_agent_obs[step] = next_agent_obs

            # Get human action if applicable
            human_action = None
            overwrite_action_flag = False
            
            if human_agent is not None:
                human_actions, overwrite_flag = human_agent.get_action(next_obs, infos["utterance"])
                actions = np.concatenate([agent_actions, human_actions.reshape(-1,1), overwrite_flag.reshape(-1,1)], axis=1)[0]
                human_action = human_actions.item()
            else:
                actions = (0, 0, 0, agent_actions.cpu().numpy().item(), 0)
                human_action = None

            next_obs, reward, terminations, truncations, info = envs.envs[0].step(actions)
            n += 1
            next_done = np.logical_or(terminations, truncations)
            total_reward += reward
            
            # Track additional metrics
            agent_action_type = info["utterance"][0]
            agent_action_length = info["utterance"][2]
            
            if agent_action_type == 2:
                type2_count += 1
                
            if overwrite_flag:
                overwritten_count += 1
                
            # Track action length frequencies
            if agent_action_length not in action_length_counts:
                action_length_counts[agent_action_length] = 0
            action_length_counts[agent_action_length] += 1
            
            # Update trajectory data with reward
            trajectory_step = {
                'step': step,
                'agent_action_type': agent_action_type,
                'agent_action': info["utterance"][1],
                'agent_action_length': agent_action_length,
                'human_action': human_action,
                'reward': reward
            }
            trajectory_data.append(trajectory_step)
            
            next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor(np.array([next_done])).to(device)
            step += 1

        return episodic_returns, episodic_type2_counts, episodic_overwritten_counts, episodic_action_length_varieties #, info_list, n_steps


class TransformerEvaluator(BaseEvaluator):
    def __init__(self, args, run_name):
        super().__init__(args, run_name)

    def evaluate(self,
        model_path: str,
        make_env: Callable,
        eval_episodes: int,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        capture_video: bool = True,
        visualize: bool = False,
    ):
        self.visualize = visualize
        
        # Create videos directory for evaluation
        os.makedirs("videos/eval", exist_ok=True)
        
        # Use RecordVideo for both wandb and non-wandb cases
        envs = gym.vector.SyncVectorEnv([make_env(self.args.env_id, 0, capture_video, self.run_name)])
        agent = model(envs, self.args).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()

        num_envs = 1
        obs = torch.zeros((self.args.num_steps, num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((self.args.num_steps, num_envs) + envs.single_action_space[-1].shape).to(device)
        rewards = torch.zeros((self.args.num_steps, num_envs)).to(device)
        env_steps = torch.zeros((self.args.num_steps, num_envs)).to(device)
        times_contexts = torch.zeros((self.args.num_steps, num_envs, self.args.context_len), dtype=torch.long).to(device)
        obs_contexts = torch.zeros((self.args.num_steps, num_envs, self.args.context_len) + envs.single_observation_space.shape).to(device)
        action_contexts = torch.zeros((self.args.num_steps, num_envs, self.args.context_len) + envs.single_action_space[-1].shape, dtype=torch.long).to(device)

        next_obs, _ = envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        env_step = torch.zeros(num_envs).to(device)

        episodic_returns = []
        # New metrics
        episodic_type2_counts = []
        episodic_overwritten_counts = []
        episodic_action_length_varieties = {}
        
        step = 0
        episode_idx = 0
        
        # Initialize episode-specific metrics
        type2_count = 0
        overwritten_count = 0
        action_length_counts = {}  # Dictionary to track frequency of each action length
        
        # Initialize trajectory data for this episode
        trajectory_data = []
        
        while len(episodic_returns) < eval_episodes:
            obs[step] = next_obs
            env_steps[step] = env_step

            if step < self.args.context_len:
                # Pre-allocate tensors with the correct shape and device
                times_context = torch.zeros((self.args.context_len, num_envs), dtype=torch.long, device=device)
                obs_context = torch.zeros((self.args.context_len, num_envs) + obs.shape[2:], dtype=torch.float32, device=device)
                action_context = torch.zeros((self.args.context_len, num_envs) + actions.shape[2:], dtype=torch.long, device=device)

                # Fill in the context more efficiently
                if step > 0:
                    times_context[:step] = self.env_steps[:step]
                    obs_context[:step] = self.obs[:step]
                    action_context[:step] = self.actions[:step]
                
                # Fill the remaining slots with the current step
                times_context[step:] = step
                obs_context[step:] = obs[step].unsqueeze(0).expand(self.args.context_len-step, -1, -1)
                action_context[step:] = actions[step].unsqueeze(0).expand(self.args.context_len-step, -1)

                times_context = times_context.transpose(0, 1)
                obs_context = obs_context.transpose(0, 1)
                action_context = action_context.transpose(0, 1)
            else:
                times_context = env_steps[step-self.args.context_len:step].transpose(0, 1)
                obs_context = obs[step-self.args.context_len:step].transpose(0, 1)
                action_context = actions[step-self.args.context_len:step].transpose(0, 1)

            times_contexts[step] = times_context.long()
            obs_contexts[step] = obs_context
            action_contexts[step] = action_context.long()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                agent_action, _, _, _ = agent.get_action_and_value([times_context.long(), obs_context, action_context.long()])
            actions[step] = agent_action
            
            # Get human action if applicable
            human_action = None
            overwrite_action_flag = False
            
            # Log trajectory data
            trajectory_step = {
                'step': step,
                'agent_action': agent_action.cpu().numpy().item(),
                'human_action': human_action,
                'overwritten': overwrite_flag,
                'reward': 0  # Will be updated after step
            }

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.envs[0].step((0.0,0.0,0.0, agent_action.cpu().numpy().item(), 0))
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor([reward]).to(device).view(-1)
            next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor([next_done]).to(device)
            env_step = torch.where(next_done==1.0, torch.zeros_like(env_step), env_step + 1)
            
            # Track additional metrics
            agent_action_type = infos["utterance"][0]
            agent_action_length = infos["utterance"][2]
            
            if agent_action_type == 2:
                type2_count += 1
                
            if overwrite_flag:
                overwritten_count += 1
                
            # Track action length frequencies
            if agent_action_length not in action_length_counts:
                action_length_counts[agent_action_length] = 0
            action_length_counts[agent_action_length] += 1
            
            # Update trajectory data with reward
            trajectory_step['reward'] = reward
            trajectory_step['agent_action_type'] = agent_action_type
            trajectory_step['agent_action_length'] = agent_action_length
            trajectory_data.append(trajectory_step)
                    
            if next_done:
                episodic_returns += [rewards[step].sum().item()]
                # Add the new metrics for this episode
                episodic_type2_counts += [type2_count]
                episodic_overwritten_counts += [overwritten_count]
                for key, value in action_length_counts.items():
                    if key not in episodic_action_length_varieties:
                        episodic_action_length_varieties[key] = []
                    episodic_action_length_varieties[key] += [value]
                
                # Visualize trajectory if enabled
                if self.visualize and len(trajectory_data) > 0:
                    self.visualize_trajectory(episode_idx, trajectory_data)
                    episode_idx += 1
                
                obs = torch.zeros((self.args.num_steps, num_envs) + envs.single_observation_space.shape).to(device)
                actions = torch.zeros((self.args.num_steps, num_envs) + envs.single_action_space[-1].shape).to(device)
                rewards = torch.zeros((self.args.num_steps, num_envs)).to(device)
                env_steps = torch.zeros((self.args.num_steps, num_envs)).to(device)
                times_contexts = torch.zeros((self.args.num_steps, num_envs, self.args.context_len), dtype=torch.long).to(device)
                obs_contexts = torch.zeros((self.args.num_steps, num_envs, self.args.context_len) + envs.single_observation_space.shape).to(device)
                action_contexts = torch.zeros((self.args.num_steps, num_envs, self.args.context_len) + envs.single_action_space[-1].shape, dtype=torch.long).to(device)

                next_obs, _ = envs.reset()
                next_obs = torch.Tensor(next_obs).to(device)
                env_step = torch.zeros(num_envs).to(device)

                step = 0
                
                # Initialize trajectory data for new episode
                trajectory_data = []
                # Reset episode-specific metrics
                type2_count = 0
                overwritten_count = 0
                action_length_counts = {}  # Reset action length counts for new episode
            else:
                step += 1

        return episodic_returns, episodic_type2_counts, episodic_overwritten_counts, episodic_action_length_varieties


class BaseBlockingEvaluator(BaseEvaluator):
    def __init__(self, args, run_name):
        super().__init__(args, run_name)

    def evaluate(self,
        model_path: str,
        make_env: Callable,
        eval_episodes: int,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        capture_video: bool = True,
        visualize: bool = False,
    ):
        self.visualize = visualize
        
        envs = gym.vector.SyncVectorEnv([make_env(self.args.env_id, 0, capture_video, self.run_name)])
        agent = model(envs, self.args).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()

        self.args.num_envs = 1 # Override num_envs to 1 for evaluation
        self.args.device = device
        self.next_agent_obs = torch.zeros((600, self.args.num_envs) + (agent.single_observation_space,)).to(device)
        self.full_next_agent_obs = torch.zeros((600*10, self.args.num_envs) + (agent.single_observation_space,)).to(device)
        human_agent = None
        if self.args.human_agent_type is not None and self.args.human_agent_type != "None":
            human_agent = HumanAgent(envs, self.args, device)

        obs, infos = envs.reset()
        episodic_returns = []
        # New metrics
        episodic_type2_counts = []
        episodic_overwritten_counts = []
        episodic_action_length_varieties = {}
        
        total_reward = 0
        # Initialize episode-specific metrics
        type2_count = 0
        overwritten_count = 0
        action_length_counts = {}  # Dictionary to track frequency of each action length
        
        step = 0
        episode_idx = 0
        total_steps = 0

        while len(episodic_returns) < eval_episodes:
            # Initialize trajectory data for this episode
            if step == 0:
                trajectory_data = []
                # Reset episode-specific metrics
                type2_count = 0
                overwritten_count = 0
                action_length_counts = {}  # Reset action length counts for new episode
            
            # Get agent action
            next_agent_obs = self.compute_next_agent_obs(torch.Tensor(obs).to(device), infos)

            agent_actions, _, _, _ = agent.get_action_and_value(torch.Tensor(next_agent_obs).to(device))
            self.next_agent_obs[step] = next_agent_obs
            
            for i in range(infos["utterance"][0][2]+self.args.rollout_reward_buffer_steps):
                # Get human action if applicable
                self.full_next_agent_obs[total_steps] = self.compute_next_agent_obs(torch.Tensor(obs).to(device), infos)
                human_actions, overwrite_flag = human_agent.get_action(torch.Tensor(obs).to(device), infos["utterance"])
                actions = np.concatenate([agent_actions, human_actions.reshape(-1,1), overwrite_flag.reshape(-1,1)], axis=1)[0]
                human_action = human_actions.item()
                
                # Execute action
                next_obs, reward, terminations, truncations, info = envs.envs[0].step(actions)
                infos = {k: np.array([info[k]]) for k in info}
                next_done = np.logical_or(terminations, truncations)
                total_reward += reward

                # Track additional metrics
                agent_action_type = info["utterance"][0]
                agent_action_length = info["utterance"][2]
                
                if agent_action_type == 2:
                    type2_count += 1
                    
                if overwrite_flag:
                    overwritten_count += 1
                    
                # Track action length frequencies
                if agent_action_length not in action_length_counts:
                    action_length_counts[agent_action_length] = 0
                action_length_counts[agent_action_length] += 1

                # Log trajectory data
                trajectory_step = {
                    'step': step,
                    'agent_action_type': agent_action_type,
                    'agent_action': info["utterance"][1],
                    'agent_action_length': agent_action_length,
                    'human_action': human_action,
                    'overwritten': overwrite_flag,
                    'reward': reward
                }
                
                # Update trajectory data with reward
                trajectory_data.append(trajectory_step)
            
                next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor([next_done]).to(device)

                if next_done:
                    episodic_returns += [total_reward]
                    # Add the new metrics for this episode
                    episodic_type2_counts += [type2_count]
                    episodic_overwritten_counts += [overwritten_count]
                    for key, value in action_length_counts.items():
                        if key not in episodic_action_length_varieties:
                            episodic_action_length_varieties[key] = []
                        episodic_action_length_varieties[key] += [value]
                    
                    # Visualize trajectory if enabled
                    if self.visualize:
                        self.visualize_trajectory(episode_idx, trajectory_data)
                    
                    obs, infos = envs.reset()
                    next_obs = torch.Tensor(obs).to(device)
                    total_reward = 0
                    step = 0
                    episode_idx += 1
                    if human_agent is not None:
                        human_agent.reset()
                else:
                    step += 1
                    
                obs = next_obs
                agent_actions = np.array([[1,0,0]]* self.args.num_envs)

        return episodic_returns, episodic_type2_counts, episodic_overwritten_counts, episodic_action_length_varieties
    
class HeuristicEvaluator(BaseEvaluator):
    def __init__(self, args, run_name):
        super().__init__(args, run_name)

    def evaluate(self,
        make_env: Callable,
        eval_episodes: int,
        device: torch.device = torch.device("cpu"),
        capture_video: bool = True,
        visualize: bool = False,
    ):
        self.visualize = visualize
        
        envs = gym.vector.SyncVectorEnv([make_env(self.args.env_id, 0, capture_video, self.run_name)])
        agent = HeuristicAgent(envs, self.args)

        self.args.num_envs = 1 # Override num_envs to 1 for evaluation
        self.args.device = device
        self.single_observation_space = (np.array(envs.single_observation_space.shape).prod() + (envs.single_action_space.shape[0]-1))*self.args.human_utterance_memory_length
        self.next_agent_obs = torch.zeros((600, self.args.num_envs) + (self.single_observation_space,)).to(device)
        self.full_next_agent_obs = torch.zeros((600*10, self.args.num_envs) + (self.single_observation_space,)).to(device)
        self.args.device = device
        human_agent = HumanAgent(envs, self.args, device)

        obs, infos = envs.reset()
        episodic_returns = []
        # New metrics
        episodic_type2_counts = []
        episodic_overwritten_counts = []
        episodic_action_length_varieties = {}
        
        total_reward = 0
        # Initialize episode-specific metrics
        type2_count = 0
        overwritten_count = 0
        action_length_counts = {}  # Dictionary to track frequency of each action length
        
        step = 0
        episode_idx = 0

        while len(episodic_returns) < eval_episodes:
            # Initialize trajectory data for this episode
            if step == 0:
                trajectory_data = []
                # Reset episode-specific metrics
                type2_count = 0
                overwritten_count = 0
                action_length_counts = {}  # Reset action length counts for new episode

            # Get human action if applicable
            next_agent_obs = torch.cat([torch.Tensor(obs).to(device), torch.Tensor(infos['utterance']).to(device)], dim=1)
            agent_actions, _, _, _ = agent.get_action_and_value(next_agent_obs)

            human_actions, overwrite_flag = human_agent.get_action(torch.Tensor(obs).to(device), infos["utterance"])
            actions = np.concatenate([agent_actions.cpu().numpy(), human_actions.reshape(-1,1), overwrite_flag.reshape(-1,1)], axis=1)[0]
            human_action = human_actions.item()
            
            # Execute action
            next_obs, reward, terminations, truncations, info = envs.envs[0].step(actions)
            infos = {k: np.array([info[k]]) for k in info}
            next_done = np.logical_or(terminations, truncations)
            total_reward += reward
            
            # Track additional metrics
            agent_action_type = info["utterance"][0]
            agent_action_length = info["utterance"][2]
            
            if agent_action_type == 2:
                type2_count += 1
                
            if overwrite_flag:
                overwritten_count += 1
                
            # Track action length frequencies
            if agent_action_length not in action_length_counts:
                action_length_counts[agent_action_length] = 0
            action_length_counts[agent_action_length] += 1
            
            # Log trajectory data
            trajectory_step = {
                'step': step,
                'agent_action_type': agent_action_type,
                'agent_action': info["utterance"][1],
                'agent_action_length': agent_action_length,
                'human_action': human_action,
                'overwritten': overwrite_flag,
                'reward': reward,
            }

            # Update trajectory data with reward
            trajectory_data.append(trajectory_step)
            
            next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor([next_done]).to(device)
            
            if next_done:
                episodic_returns += [total_reward]
                # Add the new metrics for this episode
                episodic_type2_counts += [type2_count]
                episodic_overwritten_counts += [overwritten_count]
                for key, value in action_length_counts.items():
                    if key not in episodic_action_length_varieties:
                        episodic_action_length_varieties[key] = []
                    episodic_action_length_varieties[key] += [value]
                
                # Visualize trajectory if enabled
                if self.visualize:
                    self.visualize_trajectory(episode_idx, trajectory_data)
                
                obs, infos = envs.reset()
                next_obs = torch.Tensor(obs).to(device)
                total_reward = 0
                step = 0
                episode_idx += 1
                if human_agent is not None:
                    human_agent.reset()

                self.next_agent_obs = torch.zeros((600, self.args.num_envs) + (self.single_observation_space,)).to(device)
                self.full_next_agent_obs = torch.zeros((600*10, self.args.num_envs) + (self.single_observation_space,)).to(device)
            else:
                step += 1
            
            obs = next_obs

        return episodic_returns, episodic_type2_counts, episodic_overwritten_counts, episodic_action_length_varieties