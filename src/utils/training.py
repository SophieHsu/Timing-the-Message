import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
import time
import random

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rollouts import BaseRolloutCollector, LSTMRolloutCollector, TransformerRolloutCollector
from utils.util import make_env
from utils.evaluate import BaseEvaluator, LSTMEvaluator, TransformerEvaluator


class BaseTrainer:
    def __init__(self, agent, envs, args, writer, run_name, device):
        self.agent = agent
        self.envs = envs
        self.args = args
        self.writer = writer
        self.run_name = run_name
        self.device = device
        self.rollout_collector = BaseRolloutCollector(args, agent, envs, writer, device)
        self.evaluator = BaseEvaluator(args, run_name)

    def train(self):
        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(self.args.env_id, i, self.args.capture_video, self.run_name) for i in range(self.args.num_envs)],
        )

        optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        global_step = 0
        start_time = time.time()

        for iteration in range(1, self.args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            results = self.rollout_collector.collect_rollouts(global_step)
            global_step = results["global_step"]
            
            # flatten the batch
            b_obs = results["obs"].reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = results["logprobs"].reshape(-1)
            b_actions = results["actions"].reshape((-1,) + envs.single_action_space[-1].shape)
            b_advantages = results["advantages"].reshape(-1)
            b_returns = results["returns"].reshape(-1)
            b_values = results["values"].reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.args.track and global_step % 50000 == 0: # 1 step = 50 global steps
                torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
                torch.save(self.agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
                wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
                episodic_returns = self.evaluator.evaluate(
                    f"{wandb.run.dir}/agent.pt",
                    make_env,
                    eval_episodes=3,
                    model=self.agent.__class__,
                    device="cpu",
                    capture_video=True,
                )

                if os.path.exists(f"videos/{self.run_name}"):
                    for video_file in os.listdir(f"videos/{self.run_name}"):
                        if video_file.endswith(".mp4"):
                            wandb.log({
                                f"videos/eval_{video_file}": wandb.Video(f"videos/{self.run_name}/{video_file}")
                            }, step=global_step)
                for i in range(len(episodic_returns)):  
                    self.writer.add_scalar(f"eval/episodic_return", episodic_returns[i], global_step)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        envs.close()
        self.writer.close()


class LSTMTrainer(BaseTrainer):
    def __init__(self, agent, envs, args, writer, run_name, device):
        super().__init__(agent, envs, args, writer, run_name, device)
        self.rollout_collector = LSTMRolloutCollector(args, agent, envs, writer, device)
        self.evaluator = LSTMEvaluator(args, run_name)

    def train(self):
        """Train the agent using PPO"""
        # Seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        # Initialize optimizer
        optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        global_step = 0
        start_time = time.time()
        next_lstm_state = (
            torch.zeros(self.agent.lstm.num_layers, self.args.num_envs, self.agent.lstm.hidden_size).to(self.device),
            torch.zeros(self.agent.lstm.num_layers, self.args.num_envs, self.agent.lstm.hidden_size).to(self.device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
        
        for iteration in range(1, self.args.num_iterations + 1):
            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            
            results = self.rollout_collector.collect_rollouts(global_step, initial_lstm_state)
            global_step = results["global_step"]
            initial_lstm_state = results["initial_lstm_state"]
            next_lstm_state = results["next_lstm_state"]
            
            # flatten the batch
            b_obs = results["obs"].reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = results["logprobs"].reshape(-1)
            b_actions = results["actions"].reshape((-1,) + self.envs.single_action_space[-1].shape)
            b_dones = results["dones"].reshape(-1)
            b_advantages = results["advantages"].reshape(-1)
            b_returns = results["returns"].reshape(-1)
            b_values = results["values"].reshape(-1)

            # Optimizing the policy and value network
            assert self.args.num_envs % self.args.num_minibatches == 0
            envsperbatch = self.args.num_envs // self.args.num_minibatches
            envinds = np.arange(self.args.num_envs)
            flatinds = np.arange(self.args.batch_size).reshape(self.args.num_steps, self.args.num_envs)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, self.args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.args.track and global_step % 50000 == 0: # 1 step = 50 global steps
                torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
                torch.save(self.agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
                wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
                episodic_returns = self.evaluator.evaluate(
                    f"{wandb.run.dir}/agent.pt",
                    make_env,
                    eval_episodes=3,
                    model=self.agent.__class__,
                    device="cpu",
                    capture_video=True,
                )

                if os.path.exists(f"videos/{self.run_name}"):
                    for video_file in os.listdir(f"videos/{self.run_name}"):
                        if video_file.endswith(".mp4"):
                            wandb.log({
                                f"videos/eval_{video_file}": wandb.Video(f"videos/{self.run_name}/{video_file}")
                            }, step=global_step)
                for i in range(len(episodic_returns)):  
                    self.writer.add_scalar(f"eval/episodic_return", episodic_returns[i], global_step)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.envs.close()
        self.writer.close()


class TransformerTrainer(BaseTrainer):
    def __init__(self, agent, envs, args, writer, run_name, device):
        super().__init__(agent, envs, args, writer, run_name, device)
        self.rollout_collector = TransformerRolloutCollector(args, agent, envs, writer, device)
        self.evaluator = TransformerEvaluator(args, run_name)

    def train(self):
        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(self.args.env_id, i, self.args.capture_video, self.run_name) for i in range(self.args.num_envs)],
        )
        # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        global_step = 0
        start_time = time.time()

        for iteration in range(1, self.args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            results = self.rollout_collector.collect_rollouts()

            # flatten the batch
            b_obs = results["obs"].reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = results["logprobs"].reshape(-1)
            b_actions = results["actions"].reshape((-1,) + envs.single_action_space[-1].shape)
            b_advantages = results["advantages"].reshape(-1)
            b_returns = results["returns"].reshape(-1)
            b_values = results["values"].reshape(-1)
            b_times_contexts = results["times_contexts"].reshape(-1, self.args.context_len)
            b_obs_contexts = results["obs_contexts"].reshape(-1, self.args.context_len, *envs.single_observation_space.shape)
            b_action_contexts = results["action_contexts"].reshape(-1, self.args.context_len, *envs.single_action_space[-1].shape)

            # Optimizing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value([b_times_contexts[mb_inds], b_obs_contexts[mb_inds], b_action_contexts[mb_inds]], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    optimizer.step()

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.args.track and global_step % 50000 == 0: # 1 step = 50 global steps
                torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
                torch.save(self.agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
                wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
                episodic_returns = self.evaluator.evaluate(
                    f"{wandb.run.dir}/agent.pt",
                    make_env,
                    eval_episodes=3,
                    model=self.agent.__class__,
                    device="cpu",
                    capture_video=True,
                )

                if os.path.exists(f"videos/{self.run_name}"):
                    for video_file in os.listdir(f"videos/{self.run_name}"):
                        if video_file.endswith(".mp4"):
                            wandb.log({
                                f"videos/eval_{video_file}": wandb.Video(f"videos/{self.run_name}/{video_file}")
                            }, step=global_step)
                for i in range(len(episodic_returns)):  
                    self.writer.add_scalar(f"eval/episodic_return", episodic_returns[i], global_step)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        envs.close()
        self.writer.close()