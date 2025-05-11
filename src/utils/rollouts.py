import torch
import numpy as np
import ray
import math
import time
import os
import random
from steakhouse_ai_py.mdp.steakhouse_env import CommsSteakhouseEnv
from steakhouse_ai_py.mdp.steakhouse_mdp import SteakhouseGridworld
from steakhouse_ai_py.planners.steak_planner import SteakMediumLevelActionManager
from steakhouse_ai_py.agents.steak_agent import SteakLimitVisionHumanModel
from steakhouse_ai_py.agents.notifier_agent import LangStayAgent
from src.agents.humans import HumanChefAgent
from src.agents.lstm import NotifierLSTMAgent

# Add this at the module level, outside any class
_ray_initialized = False
_ray_debug_mode = True  # Global debug mode flag

def initialize_ray(args, ray_debug_mode=False):
    """Initialize Ray for parallel rollouts."""
    if not ray_debug_mode:
        # if not ray.is_initialized():
        #     # Check if GPUs are available
        #     num_gpus = args.num_gpus if torch.cuda.is_available() else 0
        #     if args.num_gpus > 0 and num_gpus == 0:
        #         print("Warning: GPUs requested but not available. Running with CPU only.")
                
        #     ray.init(
        #         num_cpus=args.num_cpus,
        #         num_gpus=num_gpus,
        #         ignore_reinit_error=True
        #     )
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        ray.init(runtime_env={"env_vars": {"PYTHONPATH": project_root}})  # Start Ray
    else:
        global _ray_debug_mode
        _ray_debug_mode = True
        print("Running in debug mode without Ray initialization")

class BaseRolloutCollector:
    def __init__(self, args, agent, envs, writer, device, human_agent, agent_single_action_space=None, num_envs=None):
        self.args = args
        self.agent = agent
        self.envs = envs
        self.writer = writer
        self.device = device
        self.human_agent = human_agent  
        self.num_envs = num_envs if num_envs is not None else args.num_envs
        self.agent_single_action_space = agent_single_action_space
        self.initialize_storage()

    def initialize_storage(self):
        observation_space_shape = self.envs.single_observation_space if isinstance(self.envs.single_observation_space, tuple) else self.envs.single_observation_space.shape
        self.obs = torch.zeros((self.args.num_steps, self.num_envs) + observation_space_shape).to(self.device)
        self.next_agent_obs = torch.zeros((self.args.num_steps, self.num_envs) + (self.agent.single_observation_space,)).to(self.device)
        self.full_next_agent_obs = torch.zeros((self.args.num_steps*self.args.human_utterance_memory_length, self.num_envs) + (self.agent.single_observation_space,)).to(self.device)
        self.actions = torch.zeros((self.args.num_steps, self.num_envs) + self.agent_single_action_space).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.num_envs)).to(self.device) 
        self.values = torch.zeros((self.args.num_steps, self.num_envs)).to(self.device)

    def compute_next_agent_obs(self, next_obs, infos, num_envs=None, prev_agent_obs=None):
        num_envs = num_envs if num_envs is not None else self.num_envs
        if self.human_agent is not None:
            if self.args.agent_obs_mode == "history":
                # Reshape next_obs to match the expected dimensions
                next_obs_reshaped = next_obs.reshape(num_envs, -1)
                # Convert utterance to tensor and reshape
                utterance_tensor = torch.Tensor(infos['utterance']).to(self.device)
                # Concatenate along the feature dimension
                curr_agent_obs = torch.cat([next_obs_reshaped, utterance_tensor], dim=1)
                # Get previous observations
                prev_agent_obs = self.full_next_agent_obs[-1].reshape(num_envs, self.args.human_utterance_memory_length, -1)[:,1:] if prev_agent_obs is None else prev_agent_obs
                # Concatenate with current observation
                next_agent_obs = torch.cat([prev_agent_obs, curr_agent_obs.unsqueeze(1)], dim=1).reshape(num_envs, -1)
            else:
                # For non-history mode, just concatenate along the feature dimension
                next_obs_reshaped = next_obs.reshape(num_envs, -1)
                utterance_tensor = torch.Tensor(infos['utterance']).to(self.device)
                next_agent_obs = torch.cat([next_obs_reshaped, utterance_tensor], dim=1)
        else:
            next_agent_obs = next_obs
        return next_agent_obs

    def collect_rollouts(self, global_step):
        next_obs, infos = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        
        next_done = torch.zeros(self.num_envs).to(self.device)

        for step in range(0, self.args.num_steps):
            global_step += self.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if self.human_agent is not None:
                    next_agent_obs = self.compute_next_agent_obs(next_obs, infos)
                else:
                    next_agent_obs = next_obs
                self.next_agent_obs[step] = next_agent_obs
                self.full_next_agent_obs[step] = next_agent_obs
                action, logprob, _, value = self.agent.get_action_and_value(next_agent_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            if self.human_agent is not None:
                human_action, overwrite_flag = self.human_agent.get_action(next_obs, infos['utterance'])
                full_actions = np.concatenate([action.cpu().numpy(), human_action.reshape(-1, 1), overwrite_flag.reshape(-1, 1)], axis=1)
            else:
                # Create action array more efficiently
                action_np = action.cpu().numpy()
                # Pre-allocate the full action array with zeros
                full_actions = np.zeros((self.num_envs, self.envs.envs[0].unwrapped.noti_action_length+2), dtype=np.float32)
                # Only set the last element (the actual action)
                full_actions[:, self.envs.envs[0].unwrapped.noti_action_length] = action_np.reshape(-1)
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = self.envs.step(full_actions)
            # stages, new_reward, prev_shapings, dist_threshold = reward_wrapper(next_obs, stages, prev_shapings, mode=self.args.reward_mode, dist_threshold= dist_threshold)
            # filtered_reward = [value if value in filter_set else 0 for value in reward]
            # reward += filtered_reward

            next_done = np.logical_or(terminations, truncations)
            
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

            # Log episode data more efficiently
            if next_done.any():
                # Get indices of done episodes
                done_indices = torch.where(next_done)[0]
                for i in done_indices:
                    i = i.item()  # Convert to Python int for indexing
                    # print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                    self.writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                    self.writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_agent_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        results = {
            "global_step": global_step,
            "obs": self.obs,
            "next_agent_obs": self.next_agent_obs,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "returns": returns,
            "advantages": advantages,
        }

        return results

class LSTMRolloutCollector(BaseRolloutCollector):
    def __init__(self, args, agent, envs, writer, device, human_agent, agent_single_action_space, num_envs=None):
        super().__init__(args, agent, envs, writer, device, human_agent, agent_single_action_space, num_envs=num_envs)
        
    def collect_rollouts(self, global_step, initial_lstm_state):
        # TRY NOT TO MODIFY: start the game
        next_obs, infos = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        next_lstm_state = (
            torch.zeros(self.agent.lstm.num_layers, self.num_envs, self.agent.lstm.hidden_size).to(self.device),
            torch.zeros(self.agent.lstm.num_layers, self.num_envs, self.agent.lstm.hidden_size).to(self.device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
        
        for step in range(0, self.args.num_steps):
            global_step += self.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if self.human_agent is not None:
                    next_agent_obs = self.compute_next_agent_obs(next_obs, infos)
                else:
                    next_agent_obs = next_obs
                action, logprob, _, value, next_lstm_state = self.agent.get_action_and_value(next_agent_obs, next_lstm_state, next_done)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.next_agent_obs[step] = next_agent_obs
            self.full_next_agent_obs[step] = next_agent_obs
            # TRY NOT TO MODIFY: execute the game and log data.

            if self.human_agent is not None:
                human_action, overwrite_flag = self.human_agent.get_action(next_obs, infos['utterance'])
                full_actions = np.concatenate([action.cpu().numpy(), human_action.reshape(-1, 1), overwrite_flag.reshape(-1, 1)], axis=1)
            else:
                # Create action array more efficiently
                action_np = action.cpu().numpy()
                # Pre-allocate the full action array with zeros
                full_actions = np.zeros((self.num_envs, self.envs.envs[0].unwrapped.human_action_idx+1), dtype=np.float32)
                # Only set the last element (the actual action)
                full_actions[:, self.envs.envs[0].unwrapped.human_action_idx] = action_np.reshape(-1)
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = self.envs.step(full_actions)
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

            # Log episode data more efficiently
            if next_done.any():
                # Get indices of done episodes
                done_indices = torch.where(next_done)[0]
                for i in done_indices:
                    i = i.item()  # Convert to Python int for indexing
                    # print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                    self.writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                    self.writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(
                next_agent_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        results = {
            "global_step": global_step,
            "obs": self.obs,
            "next_agent_obs": self.next_agent_obs,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "returns": returns,
            "advantages": advantages,
            "initial_lstm_state": initial_lstm_state,
            "next_lstm_state": next_lstm_state,
        }

        return results

class TransformerRolloutCollector(BaseRolloutCollector):
    def __init__(self, args, agent, envs, writer, device, human_agent, agent_single_action_space):
        super().__init__(args, agent, envs, writer, device, human_agent, agent_single_action_space)

    def initialize_storage(self):
        super().initialize_storage()
        self.env_steps = torch.zeros((self.args.num_steps, self.num_envs)).to(self.device)
        self.times_contexts = torch.zeros((self.args.num_steps, self.num_envs, self.args.context_len), dtype=torch.long).to(self.device)
        self.obs_contexts = torch.zeros((self.args.num_steps, self.num_envs, self.args.context_len) + self.envs.single_observation_space.shape).to(self.device)
        self.action_contexts = torch.zeros((self.args.num_steps, self.num_envs, self.args.context_len) + self.envs.single_action_space[-1].shape, dtype=torch.long).to(self.device)
        self.full_actions = torch.zeros((self.args.num_steps, self.num_envs, 4), dtype=torch.float32).to(self.device)

    def collect_rollouts(self, global_step):
        # TRY NOT TO MODIFY: start the game
        next_obs, infos = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        env_step = torch.zeros(self.num_envs).to(self.device)


        for step in range(0, self.args.num_steps):
            global_step += self.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done
            self.env_steps[step] = env_step

            if step < self.args.context_len:
                # Pre-allocate tensors with the correct shape and device
                times_context = torch.zeros((self.args.context_len, self.num_envs), dtype=torch.long, device=self.device)
                obs_context = torch.zeros((self.args.context_len, self.num_envs) + self.obs.shape[2:], dtype=torch.float32, device=self.device)
                action_context = torch.zeros((self.args.context_len, self.num_envs) + self.actions.shape[2:], dtype=torch.long, device=self.device)

                # Fill in the context more efficiently
                if step > 0:
                    times_context[:step] = self.env_steps[:step]
                    obs_context[:step] = self.obs[:step]
                    action_context[:step] = self.actions[:step]
                
                # Fill the remaining slots with the current step
                times_context[step:] = step
                obs_context[step:] = self.obs[step].unsqueeze(0).expand(self.args.context_len-step, -1, -1)
                action_context[step:] = self.actions[step].unsqueeze(0).expand(self.args.context_len-step, -1)

                # Transpose once at the end
                times_context = times_context.transpose(0, 1)
                obs_context = obs_context.transpose(0, 1)
                action_context = action_context.transpose(0, 1)
            else:
                # Use direct slicing for better efficiency
                times_context = self.env_steps[step-self.args.context_len:step].transpose(0, 1)
                obs_context = self.obs[step-self.args.context_len:step].transpose(0, 1)
                action_context = self.actions[step-self.args.context_len:step].transpose(0, 1)

            # Ensure all tensors have the same sequence length
            assert times_context.size(1) == obs_context.size(1) == action_context.size(1), \
                f"Sequence lengths don't match: times={times_context.size(1)}, obs={obs_context.size(1)}, actions={action_context.size(1)}"

            # Store the context tensors
            self.times_contexts[step] = times_context.long()
            self.obs_contexts[step] = obs_context
            self.action_contexts[step] = action_context.long()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value([times_context.long(), obs_context, action_context.long()])
                self.values[step] = value.reshape(-1, self.num_envs)
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # Create action array more efficiently
            action_np = action.cpu().numpy()
            # Pre-allocate the full action array with zeros
            full_actions = np.zeros((self.num_envs, self.envs.envs[0].unwrapped.human_action_idx+1), dtype=np.float32)
            # Only set the last element (the actual action)
            full_actions[:, self.envs.envs[0].unwrapped.human_action_idx] = action_np.reshape(-1)
            
            # Execute environment step
            next_obs, reward, terminations, truncations, infos = self.envs.step(full_actions)
            
            # Process done flags more efficiently
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
            self.env_steps = torch.where(next_done==1.0, torch.zeros_like(self.env_steps), self.env_steps + 1)

            # Log episode data more efficiently
            if next_done.any():
                # Get indices of done episodes
                done_indices = torch.where(next_done)[0]
                for i in done_indices:
                    i = i.item()  # Convert to Python int for indexing
                    # print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                    self.writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                    self.writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value([self.times_contexts[-1], self.obs_contexts[-1], self.action_contexts[-1]]).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        results = {
            "global_step": global_step,
            "obs": self.obs,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "returns": returns,
            "advantages": advantages,
            "times_contexts": self.times_contexts,
            "obs_contexts": self.obs_contexts,
            "action_contexts": self.action_contexts,
        }

        return results

class HeuristicRolloutCollector(BaseRolloutCollector):
    def __init__(self, args, agent, envs, writer, device, human_agent, agent_single_action_space):
        super().__init__(args, agent, envs, writer, device, human_agent, agent_single_action_space)
        
    def collect_rollouts(self, global_step):
        """Collect rollouts using the heuristic agent"""
        # Reset environment
        next_obs, infos = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        for step in range(0, self.args.num_steps):
            global_step += self.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # Get action from heuristic agent
            with torch.no_grad():
                if self.human_agent is not None:
                    next_agent_obs = self.compute_next_agent_obs(next_obs, infos)
                else:
                    next_agent_obs = next_obs
                action, _, _, value = self.agent.get_action_and_value(next_agent_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.next_agent_obs[step] = next_agent_obs
            self.full_next_agent_obs[step] = next_agent_obs

            # Execute action in environment
            if self.human_agent is not None:
                human_action, overwrite_flag = self.human_agent.get_action(next_obs, infos['utterance'])
                full_actions = np.concatenate([action.cpu().numpy(), human_action.reshape(-1, 1), overwrite_flag.reshape(-1, 1)], axis=1)
            else:
                action_np = action.cpu().numpy()
                full_actions = np.zeros((self.num_envs, 5), dtype=np.float32)
                full_actions[:, 3] = action_np.reshape(-1)
            
            next_obs, reward, terminations, truncations, infos = self.envs.step(full_actions)
            next_done = np.logical_or(terminations, truncations)
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

            # Log episode data
            if next_done.any():
                done_indices = torch.where(next_done)[0]
                for i in done_indices:
                    i = i.item()
                    self.writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                    self.writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

        # Compute returns and advantages (not really needed for heuristic agent but kept for compatibility)
        with torch.no_grad():
            next_value = self.agent.get_value(next_agent_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        results = {
            "global_step": global_step,
            "obs": self.obs,
            "next_agent_obs": self.next_agent_obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "returns": returns,
            "advantages": advantages,
        }

        return results

class BaseBlockingRolloutCollector(BaseRolloutCollector):
    def __init__(self, args, agent, envs, writer, device, human_agent, agent_single_action_space):
        super().__init__(args, agent, envs, writer, device, human_agent, agent_single_action_space)

    def collect_rollouts(self, global_step):
        next_obs, infos = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        # Track the latest observations and infos for each environment
        latest_obs = next_obs.clone()
        latest_infos = infos
        
        # Store the full observation history for each environment
        # This will be used by compute_next_agent_obs
        self.obs_history = [next_obs.clone() for _ in range(self.num_envs)]
        self.infos_history = [infos for _ in range(self.num_envs)]
        total_steps = 0

        accumulated_reward = torch.zeros(self.num_envs).to(self.device)
        freeze_accumulated_reward = torch.zeros(self.num_envs).to(self.device)
        freeze_done = torch.zeros(self.num_envs).to(self.device)
        freezed_id = torch.where(freeze_accumulated_reward != 0)[0]
        
        for step in range(0, self.args.num_steps//5):
            global_step += self.num_envs
            self.dones[step] = next_done
            self.dones[step][freezed_id] = freeze_done[freezed_id]

            # ALGO LOGIC: action logic
            with torch.no_grad():
                next_agent_obs = self.compute_next_agent_obs(next_obs, infos)
                action, logprob, _, value = self.agent.get_action_and_value(next_agent_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.next_agent_obs[step] = next_agent_obs

            # Calculate the number of steps to accumulate rewards for each environment
            # action[-1] is the last element of the action vector for each environment
            steps_to_accumulate = (action[:, -1] * 2 + 3).cpu().numpy() + self.args.rollout_reward_buffer_steps
            max_steps = int(np.max(steps_to_accumulate))

            accumulated_reward = torch.zeros(self.num_envs).to(self.device)
            freeze_accumulated_reward = torch.zeros(self.num_envs).to(self.device)
            freeze_done = torch.zeros(self.num_envs).to(self.device)
            
            for i in range(max_steps):
                self.full_next_agent_obs[total_steps] = self.compute_next_agent_obs(next_obs, infos)
                # TRY NOT TO MODIFY: execute the game and log data.
                human_action, overwrite_flag = self.human_agent.get_action(latest_obs, latest_infos['utterance'])
                full_actions = np.concatenate([action.cpu().numpy(), human_action.reshape(-1, 1), overwrite_flag.reshape(-1, 1)], axis=1)
                
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(full_actions)
                
                # Update latest observations and infos
                latest_obs = torch.Tensor(next_obs).to(self.device)
                latest_infos = infos
                
                # Update the observation history for each environment
                for env_idx in range(self.num_envs):
                    # If the environment is done, reset its history
                    if next_done[env_idx]:
                        self.obs_history[env_idx] = latest_obs[env_idx].unsqueeze(0)
                        self.infos_history[env_idx] = {k: v[env_idx] if isinstance(v, np.ndarray) else v for k, v in latest_infos.items()}
                    else:
                        # Append the new observation to the history
                        self.obs_history[env_idx] = torch.cat([self.obs_history[env_idx], latest_obs[env_idx].unsqueeze(0)], dim=0)
                        # Update the infos history (this is a simplified approach, adjust as needed)
                        self.infos_history[env_idx] = {k: v[env_idx] if isinstance(v, np.ndarray) else v for k, v in latest_infos.items()}

                next_done = np.logical_or(terminations, truncations)
                
                accumulated_reward += torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                # Log episode data more efficiently
                if next_done.any():
                    # Get indices of done episodes
                    done_indices = torch.where(next_done)[0]
                    for i in done_indices:
                        i = i.item()  # Convert to Python int for indexing
                        self.writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                        self.writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

                    freeze_accumulated_reward[done_indices] = accumulated_reward[done_indices]
                    freeze_done[done_indices] = next_done[done_indices]

                # Check if we've reached the required number of steps for each environment
                complete_indices = torch.where(torch.tensor(i >= steps_to_accumulate - 1).to(self.device))[0]
                freeze_accumulated_reward[complete_indices] = accumulated_reward[complete_indices]
                total_steps += 1

                action = torch.tensor(np.array([[1,0,0]]* self.num_envs), dtype=torch.long).to(self.device) 
                    
            freezed_id = torch.where(freeze_accumulated_reward != 0)[0]
            self.rewards[step] = accumulated_reward
            self.rewards[step][freezed_id] = freeze_accumulated_reward[freezed_id]

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_agent_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        results = {
            "global_step": global_step,
            "obs": self.obs,
            "next_agent_obs": self.next_agent_obs,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "returns": returns,
            "advantages": advantages,
        }

        return results

# Define a function to conditionally apply the ray.remote decorator
def conditional_ray_remote(debug_mode):
    def decorator(cls):
    #     if not debug_mode:
    #         return ray.remote(num_gpus=0.1)
    #     return cls
    # return decorator
        if not debug_mode:
            return ray.remote(num_gpus=0.1)
        return cls
    return decorator

@conditional_ray_remote(_ray_debug_mode)
# @ray.remote(num_gpus=0.1)
class CookingLSTMRolloutWorker:
    def __init__(self, args, ray_debug_mode=False, writer=None):
        self.args = args
        self.num_envs = 1
        self.ray_debug_mode = ray_debug_mode
        
        # Ensure device is properly set
        if args.device is not None and 'cuda' in str(args.device) and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Using CPU instead.")
            self.device = torch.device('cpu')
        else:
            self.device = args.device

        # Initialize writer for logging
        self.writer = writer

    def set_agent(self, agent_state_dict):
        if self.args.layout_random:
            self.layout_name = self.args.layout_name + str(random.randint(1,5))
        else:
            self.layout_name = self.args.layout_name

        try:
            self.world_mdp = SteakhouseGridworld.from_layout_name(self.layout_name)
        except Exception as e:
            print(e)
            print(f"Layout {self.layout_name}")
            print(f"Try again...")
            if self.args.layout_random:
                self.layout_name = self.args.layout_name + str(random.randint(1,5))
            else:
                self.layout_name = self.args.layout_name
            print(f"New Layout {self.layout_name}")
            self.world_mdp = SteakhouseGridworld.from_layout_name(self.layout_name)
            print(f"Success")

        self.random_start_state_fn = self.world_mdp.get_random_objects_start_state_fn(
                random_start_pos=True,
                rnd_obj_prob_thresh=0.5
            )
        
        rand_num = random.random()
        if rand_num <= 0.01:
            self.random_start_state_fn = self.world_mdp.get_fixed_objects_start_state_fn1()
        elif rand_num <= 0.02 and rand_num > 0.01:
            self.random_start_state_fn = self.world_mdp.get_fixed_objects_start_state_fn2()
        
        self.env = CommsSteakhouseEnv.from_mdp(self.world_mdp, horizon=self.args.max_episode_steps, discretization=self.args.discretization, start_state_fn=self.random_start_state_fn)
        # self.env.reset(rand_start=self.args.rand_start)
        if self.args.one_dim_obs:
            self.single_observation_space = (self.args.steakhouse_one_dim_obs_dim,)
        else:
            self.single_observation_space = tuple(list(self.env.mdp.shape) + [23])
        self.single_action_space_n = self.env.single_action_space
        self.agent = None  # will be set via set_agent
        self.mlam = SteakMediumLevelActionManager.from_pickle_or_compute(
            self.world_mdp,
            {
                'start_orientations': True,
                'wait_allowed': True,
                'counter_goals': [],
                'counter_drop': self.world_mdp.terrain_pos_dict['X'],
                'counter_pickup': self.world_mdp.terrain_pos_dict['X'],
                'same_motion_goals': True,
                "enable_same_cell": True,
            },
            custom_filename=None,
            force_compute=False,
            info=False,
        )

        self.env._mlam = self.mlam
        self.human_agent = HumanChefAgent(
            envs=self.env,
            args=self.args,
            device=self.device,
        )
        self.human_agent.steakhouse_planner.set_agent_index(0)
        self.human_agent.steakhouse_planner.init_knowledge_base(self.world_mdp.get_standard_start_state())
        self.human_agent.steakhouse_planner.set_mdp(self.env.mdp)

        # Build a new subtask planner and HRLModel using the provided parameters.
        notifier_model = NotifierLSTMAgent(
            args=self.args,
            single_observation_space=self.single_observation_space,
            single_action_space=self.single_action_space_n,
        )
        notifier_model.load_state_dict(agent_state_dict)
        notifier_model.eval()


        notifier_human_model = SteakLimitVisionHumanModel(self.mlam, self.env.state, vision_limit=True, vision_mode="grid", vision_bound=120, kb_update_delay=0, kb_ackn_prob=False, drop_on_counter=True)
        notifier_human_model.set_agent_index(0)
        notifier_human_model.init_knowledge_base(self.world_mdp.get_standard_start_state())
        notifier_human_model.set_mdp(self.env.mdp)
        
        # Check if CUDA is available before moving to device
        if 'cuda' in str(self.device) and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Using CPU instead.")
            self.device = torch.device('cpu')
        
        notifier_model.to(self.device)

        self.agent = LangStayAgent(
            mlam=self.mlam,
            start_state=self.env.state,
            notifier_model=notifier_model,
            auto_unstuck=True,
            explore=self.args.EXPLORE,
            vision_limit=self.args.VISION_LIMIT,
            vision_bound=0,
            kb_update_delay=0,
            kb_ackn_prob=False,
            obs_size=math.prod(self.single_observation_space),
            action_size=self.single_action_space_n,
            one_dim_obs=self.args.one_dim_obs,
            drop_on_counter=self.args.drop_on_counter,
            debug=self.args.debug,
            device=self.device,
        )
        self.agent.set_agent_index(1)
        self.agent.init_knowledge_base(self.env.state)
        self.agent.set_mdp(self.env.mdp)
        self.agent.human_model = notifier_human_model

    def rollout(self, global_step, next_lstm_state):
        """
        Run a rollout for args.num_steps steps and return rollout data.
        Uses the agent's internal recurrent state.
        """
        if self.agent is None:
            raise ValueError("Agent not set. Call set_agent first.")

        args = self.args
        device = self.device
        
        # Double-check device before starting rollout
        if 'cuda' in str(device) and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available in rollout. Using CPU instead.")
            device = torch.device('cpu')
            self.device = device

        # Initialize observation using agent's get_obs; also reset the agent's recurrent states.
        vstates = self.agent.get_obs(self.env.state)
        next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
        next_done = torch.zeros(1).to(device)
        info = {}
        info["utterance"] = np.array([0]*self.env.noti_action_length)

        total_reward = 0
        first_flag = True

        # Preallocate storage for rollout data.
        obs = torch.zeros((args.num_steps,) + self.single_observation_space, device=device)
        full_next_agent_obs = torch.zeros((args.num_steps*args.human_utterance_memory_length, self.num_envs) + (self.agent.notifier_model.single_observation_space,)).to(device)
        actions = torch.zeros((args.num_steps,) + (self.single_action_space_n.shape[0]-1,), device=device)
        logprobs = torch.zeros((args.num_steps,), device=device)
        rewards = torch.zeros((args.num_steps,), device=device)
        dones = torch.zeros((args.num_steps,), device=device)
        values = torch.zeros((args.num_steps,), device=device)

        for step in range(0, self.args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if next_done:
                # Log metrics if writer is available
                if self.writer is not None:
                    self.writer.add_scalar("charts/episodic_return", total_reward, global_step)
                    self.writer.add_scalar("charts/episodic_length", step, global_step)
                
                try:
                    self.world_mdp = SteakhouseGridworld.from_layout_name(self.layout_name)
                except Exception as e:
                    print(e)
                    print(f"Layout {self.layout_name}")
                    print(f"Try again...")
                    if self.args.layout_random:
                        self.layout_name = self.args.layout_name + str(random.randint(1,5))
                    else:
                        self.layout_name = self.args.layout_name
                    print(f"New Layout {self.layout_name}")
                    self.world_mdp = SteakhouseGridworld.from_layout_name(self.layout_name)
                    print(f"Success")

                self.random_start_state_fn = self.world_mdp.get_random_objects_start_state_fn(
                        random_start_pos=True,
                        rnd_obj_prob_thresh=0.8
                    )
                
                rand_num = random.random()
                if rand_num <= 0.01:
                    self.random_start_state_fn = self.world_mdp.get_fixed_objects_start_state_fn1()
                elif rand_num <= 0.02 and rand_num > 0.01:
                    self.random_start_state_fn = self.world_mdp.get_fixed_objects_start_state_fn2()
                
                self.env = CommsSteakhouseEnv.from_mdp(self.world_mdp, horizon=self.args.max_episode_steps, discretization=self.args.discretization, start_state_fn=self.random_start_state_fn)
                self.human_agent.reset()
                self.human_agent.steakhouse_planner.init_knowledge_base(self.world_mdp.get_standard_start_state())
                self.human_agent.steakhouse_planner.set_mdp(self.env.mdp)
                self.agent.human_model.reset()
                self.agent.human_model.set_agent_index(0)
                self.agent.human_model.init_knowledge_base(self.world_mdp.get_standard_start_state())
                self.agent.human_model.set_mdp(self.env.mdp)
                self.agent.reset()
                self.agent.set_agent_index(1)
                self.agent.init_knowledge_base(self.env.state)
                self.agent.set_mdp(self.env.mdp)
                vstates = self.agent.get_obs(self.env.state)
                next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
                next_done = torch.zeros(1).to(device)

                info = {}
                info["utterance"] = np.array([0]*self.env.noti_action_length)
                first_flag = True
                total_reward = 0

            # ALGO LOGIC: action logic
            with torch.no_grad():
                infos = info
                infos["utterance"] = info["utterance"].reshape(self.num_envs, -1)
                prev_agent_obs = full_next_agent_obs[-1].reshape(self.num_envs, self.args.human_utterance_memory_length, -1)[:,1:]
                next_agent_obs = self.compute_next_agent_obs(next_obs, infos, num_envs=1, prev_agent_obs=prev_agent_obs, first_flag=first_flag)
                action, logprob, _, value, next_lstm_state = self.agent.notifier_model.get_action_and_value(next_agent_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            full_next_agent_obs[step] = next_agent_obs

            # TRY NOT TO MODIFY: execute the game and log data.
            reward = 0
            human_action, overwrite_flag = self.human_agent.get_action(self.env.state, infos['utterance'])
            full_actions = [action.cpu().numpy()[0], human_action, overwrite_flag]
            _, _ = self.agent.action(self.env.state)
            next_state, env_reward, next_done, info = self.env.step(full_actions)
            next_obs = self.agent.get_obs(next_state)
            if args.env_reward_mode:
                reward += env_reward
                reward += (sum(info["sparse_r_by_agent"]) + sum(info["shaped_r_by_agent"]))
            reward -= args.agent_step_penalty
            noti_penalty = self.agent.notification_reward(next_state, infos['utterance'][0])
            if noti_penalty < 0 and self.args.early_termination:
                next_done = True # early termination
            reward += noti_penalty * args.noti_penalty_weight
            if info["utterance"][0] == 2:
                reward -= self.args.new_noti_penalty
            # reward += (sum(info["sparse_r_by_agent"]) + sum(info["shaped_r_by_agent"]))
            total_reward += reward
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).unsqueeze(0).to(device), torch.Tensor([next_done]).to(device)

            first_flag = False

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.notifier_model.get_value(
                next_agent_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        
        if not next_done:
            # Log metrics if writer is available
            if self.writer is not None:
                self.writer.add_scalar("charts/episodic_return", total_reward, global_step)
                self.writer.add_scalar("charts/episodic_length", step, global_step)

        return {
            "global_step": global_step,
            "obs": obs,
            "next_agent_obs": full_next_agent_obs,
            "actions": actions,
            "logprobs": logprobs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "returns": returns,
            "advantages": advantages,
            "next_lstm_state": next_lstm_state,
        }

    def blocked_rollout(self, global_step, next_lstm_state):
        """
        Run a rollout for args.num_steps steps and return rollout data.
        Uses the agent's internal recurrent state.
        """
        if self.agent is None:
            raise ValueError("Agent not set. Call set_agent first.")

        args = self.args
        device = self.device
        
        # Double-check device before starting rollout
        if 'cuda' in str(device) and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available in rollout. Using CPU instead.")
            device = torch.device('cpu')
            self.device = device

        # Initialize observation using agent's get_obs; also reset the agent's recurrent states.
        vstates = self.agent.get_obs(self.env.state)
        next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
        next_done = torch.zeros(1).to(device)
        info = {}
        info["utterance"] = np.array([0]*self.env.noti_action_length)

        total_reward = 0
        first_flag = True

        # Preallocate storage for rollout data.
        obs = torch.zeros((args.num_steps,) + self.single_observation_space, device=device)
        full_next_agent_obs = torch.zeros((args.num_steps*args.human_utterance_memory_length, self.num_envs) + (self.agent.notifier_model.single_observation_space,)).to(device)
        actions = torch.zeros((args.num_steps,) + (self.single_action_space_n.shape[0]-1,), device=device)
        logprobs = torch.zeros((args.num_steps,), device=device)
        rewards = torch.zeros((args.num_steps,), device=device)
        dones = torch.zeros((args.num_steps,), device=device)
        values = torch.zeros((args.num_steps,), device=device)

        for step in range(0, self.args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if next_done:
                # Log metrics if writer is available
                if self.writer is not None:
                    self.writer.add_scalar("charts/episodic_return", total_reward, global_step)
                    self.writer.add_scalar("charts/episodic_length", step, global_step)
                
                try:
                    self.world_mdp = SteakhouseGridworld.from_layout_name(self.layout_name)
                except Exception as e:
                    print(e)
                    print(f"Layout {self.layout_name}")
                    print(f"Try again...")
                    if self.args.layout_random:
                        self.layout_name = self.args.layout_name + str(random.randint(1,5))
                    else:
                        self.layout_name = self.args.layout_name
                    print(f"New Layout {self.layout_name}")
                    self.world_mdp = SteakhouseGridworld.from_layout_name(self.layout_name)
                    print(f"Success")
                    
                if not self.args.overfit:
                    self.random_start_state_fn = self.world_mdp.get_random_objects_start_state_fn(
                            random_start_pos=True,
                            rnd_obj_prob_thresh=0.8
                        )
                    
                    rand_num = random.random()
                    if rand_num <= 0.01:
                        self.random_start_state_fn = self.world_mdp.get_fixed_objects_start_state_fn1()
                    elif rand_num <= 0.02 and rand_num > 0.01:
                        self.random_start_state_fn = self.world_mdp.get_fixed_objects_start_state_fn2()
                else:
                    self.random_start_state_fn = self.world_mdp.get_fixed_objects_start_state_fn1()
                
                self.env = CommsSteakhouseEnv.from_mdp(self.world_mdp, horizon=self.args.max_episode_steps, discretization=self.args.discretization, start_state_fn=self.random_start_state_fn)
                self.human_agent.reset()
                self.human_agent.steakhouse_planner.init_knowledge_base(self.world_mdp.get_standard_start_state())
                self.human_agent.steakhouse_planner.set_mdp(self.env.mdp)
                self.agent.human_model.reset()
                self.agent.human_model.set_agent_index(0)
                self.agent.human_model.init_knowledge_base(self.world_mdp.get_standard_start_state())
                self.agent.human_model.set_mdp(self.env.mdp)
                self.agent.reset()
                self.agent.set_agent_index(1)
                self.agent.init_knowledge_base(self.env.state)
                self.agent.set_mdp(self.env.mdp)
                vstates = self.agent.get_obs(self.env.state)
                next_obs = torch.Tensor(vstates).unsqueeze(0).to(device)
                next_done = torch.zeros(1).to(device)

                info = {}
                info["utterance"] = np.array([0]*self.env.noti_action_length)
                first_flag = True
                total_reward = 0

            # ALGO LOGIC: action logic
            with torch.no_grad():
                infos = info
                infos["utterance"] = info["utterance"].reshape(self.num_envs, -1)
                prev_agent_obs = full_next_agent_obs[-1].reshape(self.num_envs, self.args.human_utterance_memory_length, -1)[:,1:]
                next_agent_obs = self.compute_next_agent_obs(next_obs, infos, num_envs=1, prev_agent_obs=prev_agent_obs, first_flag=first_flag)
                action, logprob, _, value, next_lstm_state = self.agent.notifier_model.get_action_and_value(next_agent_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            full_next_agent_obs[step] = next_agent_obs

            reward = 0
            for j in range((action.cpu().numpy()[0][-1]*3)+2):
                # TRY NOT TO MODIFY: execute the game and log data.
                human_action, overwrite_flag = self.human_agent.get_action(self.env.state, infos['utterance'])
                if j == 0:
                    full_actions = [action.cpu().numpy()[0], human_action, overwrite_flag]
                else:
                    full_actions = [[1,0,0], human_action, overwrite_flag]
                _, _ = self.agent.action(self.env.state)
                next_state, env_reward, next_done, info = self.env.step(full_actions)
                next_obs = self.agent.get_obs(next_state)
                if args.env_reward_mode:
                    reward += env_reward
                    reward += (sum(info["sparse_r_by_agent"]) + sum(info["shaped_r_by_agent"]))
                reward -= args.agent_step_penalty
                noti_penalty = self.agent.notification_reward(next_state)
                if noti_penalty < 0 and self.args.early_termination:
                    next_done = True # early termination
                reward += noti_penalty * args.noti_penalty_weight
                if info["utterance"][0] == 2:
                    reward -= self.args.new_noti_penalty
                # reward += (sum(info["sparse_r_by_agent"]) + sum(info["shaped_r_by_agent"]))
                if next_done:
                    break
            
            next_obs, next_done = torch.Tensor(next_obs).unsqueeze(0).to(device), torch.Tensor([next_done]).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            total_reward += reward

            first_flag = False

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.notifier_model.get_value(
                next_agent_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        
        if not next_done:
            # Log metrics if writer is available
            if self.writer is not None:
                self.writer.add_scalar("charts/episodic_return", total_reward, global_step)
                self.writer.add_scalar("charts/episodic_length", step, global_step)

        return {
            "global_step": global_step,
            "obs": obs,
            "next_agent_obs": full_next_agent_obs,
            "actions": actions,
            "logprobs": logprobs,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "returns": returns,
            "advantages": advantages,
            "next_lstm_state": next_lstm_state,
        }

    def compute_next_agent_obs(self, next_obs, infos, num_envs=None, prev_agent_obs=None, first_flag=False):
        num_envs = num_envs if num_envs is not None else self.num_envs
        if self.human_agent is not None:
            if self.args.agent_obs_mode == "history":
                # Reshape next_obs to match the expected dimensions
                next_obs_reshaped = next_obs.reshape(num_envs, -1)
                # Convert utterance to tensor and reshape
                utterance_tensor = torch.Tensor(infos['utterance']).to(self.device)
                # Concatenate along the feature dimension
                curr_agent_obs = torch.cat([next_obs_reshaped, utterance_tensor], dim=1)
                # Get previous observations
                if not first_flag:
                    prev_agent_obs = self.full_next_agent_obs[-1].reshape(num_envs, self.args.human_utterance_memory_length, -1)[:,1:] if prev_agent_obs is None else prev_agent_obs
                else:
                    prev_agent_obs = curr_agent_obs.unsqueeze(1).repeat(num_envs, self.args.human_utterance_memory_length-1, 1)
                # Concatenate with current observation
                next_agent_obs = torch.cat([prev_agent_obs, curr_agent_obs.unsqueeze(1)], dim=1).reshape(num_envs, -1)
            else:
                # For non-history mode, just concatenate along the feature dimension
                next_obs_reshaped = next_obs.reshape(num_envs, -1)
                utterance_tensor = torch.Tensor(infos['utterance']).to(self.device)
                next_agent_obs = torch.cat([next_obs_reshaped, utterance_tensor], dim=1)
        else:
            next_agent_obs = next_obs
        return next_agent_obs

class CookingLSTMRolloutCollector(LSTMRolloutCollector):
    def __init__(self, args, agent, envs, writer, device, human_agent, agent_single_action_space, ray_debug_mode=False):
        super().__init__(args, agent, envs, writer, device, human_agent, agent_single_action_space, num_envs=1)
        
        # Initialize Ray if not already initialized
        initialize_ray(args, ray_debug_mode=ray_debug_mode)
        
        self.ray_debug_mode = ray_debug_mode
        
        if not self.ray_debug_mode:
            # Create a worker pool with num_envs workers
            self.workers = [CookingLSTMRolloutWorker.remote(self.args, ray_debug_mode=ray_debug_mode) for _ in range(self.args.num_envs)]
        else:
            # In debug mode, create local worker instances
            self.workers = [CookingLSTMRolloutWorker(self.args, ray_debug_mode=ray_debug_mode, writer=self.writer) for _ in range(self.args.num_envs)]
        
        # Track the last agent state to avoid unnecessary updates
        self.last_agent_state_hash = None
        
        # Set up error handling
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    def _get_agent_state_hash(self, agent_state_dict):
        """Compute a hash of the agent state to check if it has changed."""
        # Create a simple hash of the state dict
        state_str = str({k: v.sum().item() if isinstance(v, torch.Tensor) else v 
                         for k, v in agent_state_dict.items()})
        return hash(state_str)

    def _update_workers(self, agent_state_dict):
        """Update workers with the latest agent state if it has changed."""
        # current_hash = self._get_agent_state_hash(agent_state_dict)
        
        # if current_hash != self.last_agent_state_hash:
        # Agent state has changed, update all workers
        if not self.ray_debug_mode:
            ray.get([w.set_agent.remote(agent_state_dict) for w in self.workers])
        else:
            for worker in self.workers:
                worker.set_agent(agent_state_dict)
            # self.last_agent_state_haPW@toyo113800
            # sh = current_hash

    def collect_rollouts(self, global_step, next_lstm_state):
        """Collect rollouts using Ray workers with improved error handling and efficiency."""
        # Get the current agent state
        agent_state_dict = {k: v.cpu().clone() for k, v in self.agent.state_dict().items()}
        
        # Update workers only if the agent state has changed
        self._update_workers(agent_state_dict)
        
        if self.ray_debug_mode:
            # In debug mode, run rollouts locally
            rollouts = []
            for i, worker in enumerate(self.workers):
                worker_next_lstm_state= (next_lstm_state[0][:,i,:].reshape(-1,1,self.args.lstm_hidden_dim), next_lstm_state[1][:,i,:].reshape(-1,1,self.args.lstm_hidden_dim))
                try:
                    if self.args.block_rollout:
                        result = worker.blocked_rollout(global_step, worker_next_lstm_state)
                    else:
                        result = worker.rollout(global_step, worker_next_lstm_state)
                    rollouts.append(result)
                except Exception as e:
                    print(f"Error in debug mode rollout: {e}")
                    # Use fallback for this worker
                    rollouts.append(self._fallback_rollout(global_step, worker_next_lstm_state))
        else:
            # Launch rollouts asynchronously
            rollout_futures = []
            for i, worker in enumerate(self.workers):
                worker_next_lstm_state= (next_lstm_state[0][:,i,:].reshape(-1,1,self.args.lstm_hidden_dim), next_lstm_state[1][:,i,:].reshape(-1,1,self.args.lstm_hidden_dim))
                if self.args.block_rollout:
                    rollout_futures.append(worker.blocked_rollout.remote(global_step, worker_next_lstm_state))
                else:
                    rollout_futures.append(worker.rollout.remote(global_step, worker_next_lstm_state))
            
            # Collect results with error handling
            rollouts = []
            for i, future in enumerate(rollout_futures):
                retries = 0
                while retries < self.max_retries:
                    try:
                        result = ray.get(future, timeout=60)  # Add timeout to prevent hanging
                        rollouts.append(result)
                        break
                    except Exception as e:
                        retries += 1
                        if retries >= self.max_retries:
                            print(f"Worker {i} failed after {retries} retries: {e}")
                            # Create a new worker to replace the failed one
                            self.workers[i] = CookingLSTMRolloutWorker.remote(self.args, ray_debug_mode=self.ray_debug_mode)
                            # Update the new worker with the current agent state
                            self._update_workers(agent_state_dict)
                            # Try one more time with the new worker
                            worker_next_lstm_state = (next_lstm_state[0][:,i,:].reshape(-1,1,self.args.lstm_hidden_dim), next_lstm_state[1][:,i,:].reshape(-1,1,self.args.lstm_hidden_dim))
                            try:
                                if self.args.block_rollout:
                                    result = ray.get(self.workers[i].blocked_rollout.remote(global_step, worker_next_lstm_state), timeout=60)
                                else:
                                    result = ray.get(self.workers[i].rollout.remote(global_step, worker_next_lstm_state), timeout=60)
                                rollouts.append(result)
                            except Exception as e:
                                print(f"Replacement worker {i} also failed: {e}")
                                # If all else fails, use a fallback approach
                                rollouts.append(self._fallback_rollout(global_step, worker_next_lstm_state))
                        else:
                            print(f"Retrying worker {i} after error: {e}")
                            time.sleep(self.retry_delay)
        
        # Process results
        rollout_obs = torch.stack([r["obs"] for r in rollouts], dim=1)          
        rollout_next_agent_obs = torch.stack([r["next_agent_obs"] for r in rollouts], dim=1)
        rollout_actions = torch.stack([r["actions"] for r in rollouts], dim=1)    
        rollout_logprobs = torch.stack([r["logprobs"] for r in rollouts], dim=1)
        rollout_rewards = torch.stack([r["rewards"] for r in rollouts], dim=1)
        rollout_dones = torch.stack([r["dones"] for r in rollouts], dim=1)
        rollout_values = torch.stack([r["values"] for r in rollouts], dim=1)
        rollout_advantages = torch.stack([r["advantages"] for r in rollouts], dim=1)
        rollout_returns = torch.stack([r["returns"] for r in rollouts], dim=1)
        rollout_next_lstm_state = (torch.stack([r["next_lstm_state"][0][0] for r in rollouts], dim=1).to(self.device), 
                                  torch.stack([r["next_lstm_state"][1][0] for r in rollouts], dim=1).to(self.device))

        results = {
            "global_step": rollouts[0]["global_step"],
            "obs": rollout_obs,
            "next_agent_obs": rollout_next_agent_obs,
            "actions": rollout_actions,
            "logprobs": rollout_logprobs,
            "rewards": rollout_rewards,
            "dones": rollout_dones,
            "values": rollout_values,
            "returns": rollout_returns,
            "advantages": rollout_advantages,
            "next_lstm_state": rollout_next_lstm_state,
        }

        return results
    
    def _fallback_rollout(self, global_step, next_lstm_state):
        """Fallback method to run a rollout locally if all workers fail."""
        print("Using fallback rollout method")
        # This is a simplified version that runs a single rollout locally
        # In a real implementation, you would want to make this more robust
        return self.collect_rollouts_local(global_step, next_lstm_state)
    
    def collect_rollouts_local(self, global_step, next_lstm_state):
        """Run a single rollout locally as a fallback."""
        # This is a simplified version that runs a single rollout
        # In a real implementation, you would want to make this more robust
        worker = CookingLSTMRolloutWorker(self.args, self.ray_debug_mode)
        agent_state_dict = {k: v.cpu().clone() for k, v in self.agent.state_dict().items()}
        worker.set_agent(agent_state_dict)
        if self.args.block_rollout:
            return worker.blocked_rollout(global_step, next_lstm_state)
        else:
            return worker.rollout(global_step, next_lstm_state)
    
    def __del__(self):
        """Clean up Ray resources when the collector is destroyed."""
        # Note: This is a simple cleanup. In a production environment, you might want
        # to implement a more sophisticated cleanup mechanism.
        if not self.ray_debug_mode:
            try:
                # Only terminate workers, not the entire Ray cluster
                for worker in self.workers:
                    ray.kill(worker)
            except:
                pass  # Ignore errors during cleanup
