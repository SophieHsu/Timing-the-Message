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
from src.utils.rollouts import BaseRolloutCollector, LSTMRolloutCollector, TransformerRolloutCollector, HeuristicRolloutCollector, BaseBlockingRolloutCollector, CookingLSTMRolloutCollector
from src.utils.util import make_env
from src.utils.evaluate import BaseEvaluator, LSTMEvaluator, TransformerEvaluator, BaseBlockingEvaluator, CookingLSTMEvaluator


class BaseTrainer:
    def __init__(self, agent, envs, args, writer, run_name, device, human_agent):
        self.agent = agent
        self.envs = envs
        self.args = args
        self.writer = writer
        self.run_name = run_name
        self.device = device
        self.agent_single_action_space = envs.single_action_space[-1].shape if human_agent is None else envs.single_action_space[:-1].shape
        self.rollout_collector = BaseRolloutCollector(args, agent, envs, writer, device, human_agent, agent_single_action_space=self.agent_single_action_space)
        self.evaluator = BaseEvaluator(args, run_name)
        self.human_agent = human_agent

    def _log_eval_metrics(self, episodic_returns, type2_counts, overwritten_counts, action_length_varieties, global_step):
        """Helper function to log evaluation metrics to wandb."""
        # Log basic metrics
        wandb.log({
            f"eval/episodic_mean_return": np.mean(episodic_returns),
            f"eval/notify_mean_count": np.mean(type2_counts),
            f"eval/overwritten_mean_count": np.mean(overwritten_counts),
        }, step=global_step)

        # Create data for the combined action length plot
        action_length_data = []
        for key, value in action_length_varieties.items():
            mean_count = np.mean(value)
            wandb.log({
                f"eval/action_length_{key}_mean_count": mean_count
            }, step=global_step)
            action_length_data.append([int(key), mean_count])

        # Sort data by action length for consistent plotting
        action_length_data.sort(key=lambda x: x[0])

        # Create a combined line plot for all action lengths
        if len(action_length_data) > 0:
            wandb.log({
                "eval/action_length_counts": wandb.plot.line_series(
                    xs=[global_step] * len(action_length_data),
                    ys=[[d[1]] for d in action_length_data],
                    keys=[f"Length {d[0]}" for d in action_length_data],
                    title="Action Length Counts Over Time"
                )
            }, step=global_step)

    def train(self):
        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

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
            b_next_agent_obs = results["next_agent_obs"].reshape((-1,) + (self.agent.single_observation_space,))
            b_logprobs = results["logprobs"].reshape(-1)
            b_actions = results["actions"].reshape((-1,) + self.agent_single_action_space)
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

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_next_agent_obs[mb_inds], b_actions.long()[mb_inds])
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

            if self.args.track and global_step % self.args.save_freq == 0: # 1 step = 50 global steps
                torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
                torch.save(self.agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
                wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
                episodic_returns, type2_counts, overwritten_counts, action_length_varieties = self.evaluator.evaluate(
                    f"{wandb.run.dir}/agent.pt",
                    make_env,
                    eval_episodes=3,
                    model=self.agent.__class__,
                    device="cpu",
                    capture_video=True,
                    visualize=False,
                )

                if os.path.exists(f"videos/{self.run_name}"):
                    for video_file in os.listdir(f"videos/{self.run_name}"):
                        if video_file.endswith(".mp4"):
                            wandb.log({
                                f"videos/eval_{video_file}": wandb.Video(f"videos/{self.run_name}/{video_file}")
                            }, step=global_step)
                
                # Log evaluation metrics using the helper function
                self._log_eval_metrics(episodic_returns, type2_counts, overwritten_counts, action_length_varieties, global_step)

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
    def __init__(self, agent, envs, args, writer, run_name, device, human_agent):
        super().__init__(agent, envs, args, writer, run_name, device, human_agent)
        self.rollout_collector = LSTMRolloutCollector(args, agent, envs, writer, device, human_agent, agent_single_action_space=self.agent_single_action_space)
        self.evaluator = LSTMEvaluator(args, run_name)
        

    def train(self):
        """Train the agent using PPO"""
        # Seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        # Initialize optimizer
        optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        if self.args.model_run_id is not None:
            api = wandb.Api()
            run = api.run(f"{self.args.wandb_entity}/timing/{self.args.model_run_id}")
            model_path = run.config['filepath'] + "/optimizer.pt"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Optimizer file not found: {model_path}")
            else:
                optimizer.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded optimizer from {model_path}")

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
            
            results = self.rollout_collector.collect_rollouts(global_step, next_lstm_state)
            global_step = results["global_step"]
            next_lstm_state = results["next_lstm_state"]
            
            # flatten the batch
            b_next_agent_obs = results["next_agent_obs"].reshape((-1,) + (self.agent.single_observation_space,))
            b_logprobs = results["logprobs"].reshape(-1)
            b_actions = results["actions"].reshape((-1,) + self.agent_single_action_space)
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
                        b_next_agent_obs[mb_inds],
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

            if self.args.track and global_step % self.args.save_freq == 0: # 1 step = 50 global steps
                torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
                torch.save(self.agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
                wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
                for fixed_objects_start_state_mode in range(0,3):
                    try:
                        episodic_returns, type2_counts, overwritten_counts, action_length_varieties = self.evaluator.evaluate(
                            f"{wandb.run.dir}/agent.pt",
                            make_env,
                            eval_episodes=3,
                            model=self.agent.__class__,
                            device="cpu" if not torch.cuda.is_available() else "cuda",
                            capture_video=True,
                            use_random_start_state=True,
                            fixed_objects_start_state_mode=fixed_objects_start_state_mode,
                        )

                        if os.path.exists(f"videos/{self.run_name}"):
                            for video_file in os.listdir(f"videos/{self.run_name}"):
                                if video_file.endswith(".mp4"):
                                    wandb.log({
                                        f"videos/eval_{video_file}": wandb.Video(f"videos/{self.run_name}/{fixed_objects_start_state_mode}.mp4")
                                    }, step=global_step)
                        
                        # Log evaluation metrics using the helper function
                        self._log_eval_metrics(episodic_returns, type2_counts, overwritten_counts, action_length_varieties, global_step)
                    except Exception as e:
                        print(e)
                        pass

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
    def __init__(self, agent, envs, args, writer, run_name, device, human_agent):
        super().__init__(agent, envs, args, writer, run_name, device, human_agent)
        self.rollout_collector = TransformerRolloutCollector(args, agent, envs, writer, device, human_agent, agent_single_action_space=self.agent_single_action_space)
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

            results = self.rollout_collector.collect_rollouts(global_step)
            global_step = results["global_step"]
            # flatten the batch
            b_logprobs = results["logprobs"].reshape(-1)
            b_actions = results["actions"].reshape((-1,) + self.agent_single_action_space)
            b_advantages = results["advantages"].reshape(-1)
            b_returns = results["returns"].reshape(-1)
            b_values = results["values"].reshape(-1)
            b_times_contexts = results["times_contexts"].reshape(-1, self.args.context_len)
            b_obs_contexts = results["obs_contexts"].reshape(-1, self.args.context_len, *envs.single_observation_space.shape)
            b_action_contexts = results["action_contexts"].reshape(-1, self.args.context_len, *self.agent_single_action_space)

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

            if self.args.track and global_step % self.args.save_freq == 0: # 1 step = 50 global steps
                torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
                torch.save(self.agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
                wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
                episodic_returns, type2_counts, overwritten_counts, action_length_varieties = self.evaluator.evaluate(
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
                
                # Log evaluation metrics using the helper function
                self._log_eval_metrics(episodic_returns, type2_counts, overwritten_counts, action_length_varieties, global_step)

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


class HeuristicTrainer(BaseTrainer):
    def __init__(self, agent, envs, args, writer, run_name, device, human_agent):
        super().__init__(agent, envs, args, writer, run_name, device, human_agent)
        self.rollout_collector = HeuristicRolloutCollector(args, agent, envs, writer, device, human_agent, agent_single_action_space=self.agent_single_action_space)
        self.evaluator = BaseEvaluator(args, run_name)

    def train(self):
        """Train the heuristic agent - in this case, just run episodes and log results"""
        # Seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        global_step = 0
        start_time = time.time()

        for iteration in range(1, self.args.num_iterations + 1):
            # Collect rollouts using the heuristic agent
            results = self.rollout_collector.collect_rollouts(global_step)
            global_step = results["global_step"]

            # Log metrics
            if self.args.track and global_step % self.args.save_freq == 0:
                torch.save(self.agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
                
                episodic_returns, type2_counts, overwritten_counts, action_length_varieties = self.evaluator.evaluate(
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
                
                # Log evaluation metrics using the helper function
                self._log_eval_metrics(episodic_returns, type2_counts, overwritten_counts, action_length_varieties, global_step)

            # Log performance metrics
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.envs.close()
        self.writer.close()


class BlockingTrainer(BaseTrainer):
    def __init__(self, agent, envs, args, writer, run_name, device, human_agent):
        super().__init__(agent, envs, args, writer, run_name, device, human_agent)
        self.rollout_collector = BaseBlockingRolloutCollector(args, agent, envs, writer, device, human_agent, agent_single_action_space=self.agent_single_action_space)
        self.evaluator = BaseBlockingEvaluator(args, run_name)

    def train(self):
        """Train the blocking agent"""
        # Seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        global_step = 0
        start_time = time.time()

        for iteration in range(1, self.args.num_iterations + 1):
            # Collect rollouts using the blocking agent
            results = self.rollout_collector.collect_rollouts(global_step)
            global_step = results["global_step"]

            # Log metrics
            if self.args.track and global_step % self.args.save_freq == 0:
                torch.save(self.agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
                
                episodic_returns, type2_counts, overwritten_counts, action_length_varieties = self.evaluator.evaluate(
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
                
                # Log evaluation metrics using the helper function
                self._log_eval_metrics(episodic_returns, type2_counts, overwritten_counts, action_length_varieties, global_step)

            # Log performance metrics
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        self.envs.close()
        self.writer.close()
        

class CookingLSTMTrainer(LSTMTrainer):
    def __init__(self, agent, envs, args, writer, run_name, device, human_agent, ray_debug_mode=False):
        
        super().__init__(agent, envs, args, writer, run_name, device, human_agent)
        self.rollout_collector = CookingLSTMRolloutCollector(args, agent, envs, writer, device, human_agent, agent_single_action_space=self.agent_single_action_space, ray_debug_mode=ray_debug_mode)
        self.evaluator = CookingLSTMEvaluator(args, run_name)

