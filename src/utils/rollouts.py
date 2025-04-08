import torch
import numpy as np
import time

class BaseRolloutCollector:
    def __init__(self, args, agent, envs, writer, device):
        self.args = args
        self.agent = agent
        self.envs = envs
        self.writer = writer
        self.device = device
        self.initialize_storage()
    def initialize_storage(self):
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_action_space[-1].shape).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device) 
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

    def collect_rollouts(self, global_step):
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        dist_threshold = 1.5
   
        for step in range(0, self.args.num_steps):
            global_step += self.args.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # Create action array more efficiently
            action_np = action.cpu().numpy()
            # Pre-allocate the full action array with zeros
            full_actions = np.zeros((4, self.args.num_envs), dtype=np.float32)
            # Only set the last element (the actual action)
            full_actions[3] = action_np.reshape(-1)
            
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
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
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
        }

        return results

class LSTMRolloutCollector(BaseRolloutCollector):
    def __init__(self, args, agent, envs, writer, device):
        super().__init__(args, agent, envs, writer, device)
        
    def collect_rollouts(self, global_step, initial_lstm_state):
        # TRY NOT TO MODIFY: start the game
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        next_lstm_state = (
            torch.zeros(self.agent.lstm.num_layers, self.args.num_envs, self.agent.lstm.hidden_size).to(self.device),
            torch.zeros(self.agent.lstm.num_layers, self.args.num_envs, self.agent.lstm.hidden_size).to(self.device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

        for step in range(0, self.args.num_steps):
            global_step += self.args.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = self.agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # Create action array more efficiently
            action_np = action.cpu().numpy()
            # Pre-allocate the full action array with zeros
            full_actions = np.zeros((4, self.args.num_envs), dtype=np.float32)
            # Only set the last element (the actual action)
            full_actions[3] = action_np.reshape(-1)
            
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
                next_obs,
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
    def __init__(self, args, agent, envs, writer, device):
        super().__init__(args, agent, envs, writer, device)

    def initialize_storage(self):
        super().initialize_storage()
        self.env_steps = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.times_contexts = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.context_len), dtype=torch.long).to(self.device)
        self.obs_contexts = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.context_len) + self.envs.single_observation_space.shape).to(self.device)
        self.action_contexts = torch.zeros((self.args.num_steps, self.args.num_envs, self.args.context_len) + self.envs.single_action_space[-1].shape, dtype=torch.long).to(self.device)
        self.full_actions = torch.zeros((self.args.num_steps, self.args.num_envs, 4), dtype=torch.float32).to(self.device)

    def collect_rollouts(self):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        env_step = torch.zeros(self.args.num_envs).to(self.device)

        for step in range(0, self.args.num_steps):
            global_step += self.args.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done
            self.env_steps[step] = env_step

            if step < self.args.context_len:
                # Pre-allocate tensors with the correct shape and device
                times_context = torch.zeros((self.args.context_len, self.args.num_envs), dtype=torch.long, device=self.device)
                obs_context = torch.zeros((self.args.context_len, self.args.num_envs) + self.obs.shape[2:], dtype=torch.float32, device=self.device)
                action_context = torch.zeros((self.args.context_len, self.args.num_envs) + self.actions.shape[2:], dtype=torch.long, device=self.device)

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
                self.values[step] = value.reshape(-1, self.args.num_envs)
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # Create action array more efficiently
            action_np = action.cpu().numpy()
            # Pre-allocate the full action array with zeros
            full_action = np.zeros((4, self.args.num_envs), dtype=np.float32)
            # Only set the last element (the actual action)
            full_action[3] = action_np.reshape(-1)
            
            # Execute environment step
            next_obs, reward, terminations, truncations, infos = self.envs.step(full_action)
            
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