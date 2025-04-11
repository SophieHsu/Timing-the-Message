import torch
import gymnasium as gym
import numpy as np
from typing import Callable
import os
from agents.humans import HumanAgent
import copy

class BaseEvaluator:
    def __init__(self, args, run_name):
        self.args = copy.deepcopy(args)
        self.run_name = run_name

    def compute_next_agent_obs(self, next_obs, infos):
        if self.args.agent_obs_mode == "history":
            curr_agent_obs = torch.cat([next_obs, torch.Tensor(infos['utterance']).to(self.args.device)], dim=1)
            prev_agent_obs = self.next_agent_obs[-1].reshape(self.args.num_envs, self.args.human_utterance_memory_length, -1)[:,1:]
            next_agent_obs = torch.cat([prev_agent_obs, curr_agent_obs.unsqueeze(1)], dim=1).reshape(self.args.num_envs, -1)
        else:
            next_agent_obs = torch.cat([next_obs, torch.Tensor(infos['utterance']).to(self.args.device)], dim=1)

        return next_agent_obs


    def evaluate(self,
        model_path: str,
        make_env: Callable,
        eval_episodes: int,
        model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        capture_video: bool = True,
    ):
        envs = gym.vector.SyncVectorEnv([make_env(self.args.env_id, 0, capture_video, self.run_name)])
        agent = model(envs, self.args).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()

        self.args.num_envs = 1 # Override num_envs to 1 for evaluation
        self.args.device = device
        self.next_agent_obs = torch.zeros((self.args.num_steps, self.args.num_envs) + (agent.single_observation_space,)).to(device)
        human_agent = None
        if self.args.human_agent_type is not None:
            human_agent = HumanAgent(envs, self.args, device)

        obs, infos = envs.reset()
        episodic_returns = []
        total_reward = 0
        step = 0
        while len(episodic_returns) < eval_episodes:

            if human_agent is not None:
                next_agent_obs = self.compute_next_agent_obs(torch.Tensor(obs).to(device), infos)
            else:
                next_agent_obs = torch.Tensor(obs).to(device)

            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(next_agent_obs).to(device))
            self.next_agent_obs[step] = next_agent_obs

            if human_agent is not None:
                human_actions = human_agent.get_action(torch.Tensor(obs).to(device), infos["utterance"])
                actions = np.concatenate([actions, human_actions.reshape(-1,1)], axis=1)[0]
            else:
                actions = (0, 0, 0, actions.cpu().numpy().item())

            next_obs, reward, terminations, truncations, info = envs.envs[0].step(actions)
            infos = {k: np.array([info[k]]) for k in info}
            next_done = np.logical_or(terminations, truncations)
            total_reward += reward
            next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor([next_done]).to(device)

            if next_done:
                episodic_returns += [total_reward]
                obs, infos = envs.reset()
                next_obs = torch.Tensor(obs).to(device)
                total_reward = 0
                step = 0
                if human_agent is not None:
                    human_agent.reset()
            obs = next_obs

        return episodic_returns


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
    ):
        envs = gym.vector.SyncVectorEnv([make_env(self.args.env_id, 0, capture_video, self.run_name)])
        agent = model(envs, self.args).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        agent.eval()

        self.args.num_envs = 1 # Override num_envs to 1 for evaluation
        self.args.device = device
        self.next_agent_obs = torch.zeros((self.args.num_steps, self.args.num_envs) + (agent.single_observation_space,)).to(device)
        human_agent = None
        if self.args.human_agent_type is not None:
            human_agent = HumanAgent(envs, self.args, device)

        next_done = torch.zeros(1).to(device)
        total_reward = np.zeros(1)
        init_flag = True

        episodic_returns = []
        n_steps = []
        step = 0
        while len(episodic_returns) < eval_episodes:
            if next_done[0] or init_flag:
                if next_done[0]:
                    # log before moving on to next episode
                    # print(f"eval_episode={len(episodic_returns)}, episodic_return={total_reward[0]}")
                    episodic_returns += [total_reward[0]]
                    n_steps.append(n)
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

            if human_agent is not None:
                next_agent_obs = self.compute_next_agent_obs(next_obs, infos)
            else:
                next_agent_obs = next_obs
            actions, _, _, _, next_lstm_state = agent.get_action_and_value(next_agent_obs, next_lstm_state, next_done)
            self.next_agent_obs[step] = next_agent_obs
            if human_agent is not None:
                human_actions = human_agent.get_action(next_obs, infos["utterance"])
                actions = np.concatenate([actions, human_actions.reshape(-1,1)], axis=1)[0]
            else:
                actions = (0 ,0 ,0 , actions.cpu().numpy().item())

            next_obs, reward, terminations, truncations, info = envs.envs[0].step(actions)
            n += 1
            next_done = np.logical_or(terminations, truncations)
            total_reward += reward
            next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor(np.array([next_done])).to(device)

        return episodic_returns #, info_list, n_steps


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
    ):
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
        step = 0
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
                action, _, _, _ = agent.get_action_and_value([times_context.long(), obs_context, action_context.long()])
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.envs[0].step((0.0,0.0,0.0, action.cpu().numpy().item()))
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor([reward]).to(device).view(-1)
            next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor([next_done]).to(device)
            env_step = torch.where(next_done==1.0, torch.zeros_like(env_step), env_step + 1)
                    
            if next_done:
                episodic_returns += [rewards[step].sum().item()]
                
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

        return episodic_returns