# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium_envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.transformers import TransformerPolicy, TransformerCritic
from agents.eval import evaluate
@dataclass
class Args:
    exp_name: str = "ppo"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "timing"
    """the wandb's project name"""
    wandb_entity: str = "yachuanh"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "gymnasium_envs/NotiLunarLander"
    """the id of the environment"""
    total_timesteps: int = int(1e9)
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # transformer specific arguments
    context_len: int = 10
    """the context length for the transformer"""
    n_blocks: int = 4
    """the number of transformer blocks"""
    h_dim: int = 32
    """the hidden dimension for the transformer"""
    n_heads: int = 4
    """the number of attention heads for the transformer"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.critic = TransformerCritic(state_dim=np.array(envs.single_observation_space.shape).prod(), 
                                        act_dim=envs.single_action_space[-1].n, 
                                        context_len=args.context_len, 
                                        n_blocks=args.n_blocks, 
                                        h_dim=args.h_dim, 
                                        n_heads=args.n_heads, 
                                        drop_p=0.1)
        
        self.actor = TransformerPolicy(state_dim=np.array(envs.single_observation_space.shape).prod(), 
                                       act_dim=envs.single_action_space[-1].n, 
                                       n_blocks=args.n_blocks, 
                                       h_dim=args.h_dim, 
                                       context_len=args.context_len, 
                                       n_heads=args.n_heads, 
                                       drop_p=0.1)

    def get_value(self, x):
        if isinstance(x, list):
            times, states, actions = x
            return self.critic(times, states, actions)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if isinstance(x, list):
            times, states, actions = x
            logits = self.actor(times, states, actions)

            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(times, states, actions)
        else:
            logits = self.actor(x)

            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id.split('/')[-1]}_{args.exp_name}_{args.seed}_{args.learning_rate}_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space[-1].shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    env_steps = torch.zeros((args.num_steps, args.num_envs)).to(device)
    times_contexts = torch.zeros((args.num_steps, args.num_envs, args.context_len), dtype=torch.long).to(device)
    obs_contexts = torch.zeros((args.num_steps, args.num_envs, args.context_len) + envs.single_observation_space.shape).to(device)
    action_contexts = torch.zeros((args.num_steps, args.num_envs, args.context_len) + envs.single_action_space[-1].shape, dtype=torch.long).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    env_step = torch.zeros(args.num_envs).to(device)
    
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            env_steps[step] = env_step

            if step < args.context_len:
                # Pre-allocate tensors with the correct shape and device
                times_context = torch.zeros((args.context_len, args.num_envs), dtype=torch.long, device=device)
                obs_context = torch.zeros((args.context_len, args.num_envs) + obs.shape[2:], dtype=torch.float32, device=device)
                action_context = torch.zeros((args.context_len, args.num_envs) + actions.shape[2:], dtype=torch.long, device=device)

                # Fill in the context more efficiently
                if step > 0:
                    times_context[:step] = env_steps[:step]
                    obs_context[:step] = obs[:step]
                    action_context[:step] = actions[:step]
                
                # Fill the remaining slots with the current step
                times_context[step:] = step
                obs_context[step:] = obs[step].unsqueeze(0).expand(args.context_len-step, -1, -1)
                action_context[step:] = actions[step].unsqueeze(0).expand(args.context_len-step, -1)

                # Transpose once at the end
                times_context = times_context.transpose(0, 1)
                obs_context = obs_context.transpose(0, 1)
                action_context = action_context.transpose(0, 1)
            else:
                # Use direct slicing for better efficiency
                times_context = env_steps[step-args.context_len:step].transpose(0, 1)
                obs_context = obs[step-args.context_len:step].transpose(0, 1)
                action_context = actions[step-args.context_len:step].transpose(0, 1)

            # Ensure all tensors have the same sequence length
            assert times_context.size(1) == obs_context.size(1) == action_context.size(1), \
                f"Sequence lengths don't match: times={times_context.size(1)}, obs={obs_context.size(1)}, actions={action_context.size(1)}"

            # Store the context tensors
            times_contexts[step] = times_context.long()
            obs_contexts[step] = obs_context
            action_contexts[step] = action_context.long()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value([times_context.long(), obs_context, action_context.long()])
                values[step] = value.reshape(-1, args.num_envs)
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # Create action array more efficiently
            action_np = action.cpu().numpy()
            # Pre-allocate the full action array with zeros
            full_actions = np.zeros((4, args.num_envs), dtype=np.float32)
            # Only set the last element (the actual action)
            full_actions[3] = action_np.reshape(-1)
            
            # Execute environment step
            next_obs, reward, terminations, truncations, infos = envs.step(full_actions)
            
            # Process done flags more efficiently
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            env_step = torch.where(next_done==1.0, torch.zeros_like(env_step), env_step + 1)

            # Log episode data more efficiently
            if next_done.any():
                # Get indices of done episodes
                done_indices = torch.where(next_done)[0]
                for i in done_indices:
                    i = i.item()  # Convert to Python int for indexing
                    print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], global_step)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value([times_contexts[-1], obs_contexts[-1], action_contexts[-1]]).reshape(1, -1)
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space[-1].shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_times_contexts = times_contexts.reshape(-1, args.context_len)
        b_obs_contexts = obs_contexts.reshape(-1, args.context_len, *envs.single_observation_space.shape)
        b_action_contexts = action_contexts.reshape(-1, args.context_len, *envs.single_action_space[-1].shape)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value([b_times_contexts[mb_inds], b_obs_contexts[mb_inds], b_action_contexts[mb_inds]], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track and global_step % 50000 == 0: # 1 step = 50 global steps
            torch.save(optimizer.state_dict(), f"{wandb.run.dir}/optimizer.pt")
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"{wandb.run.dir}/optimizer.pt", base_path=wandb.run.dir, policy="now")
            wandb.save(f"{wandb.run.dir}/agent.pt", base_path=wandb.run.dir, policy="now")
            episodic_returns = evaluate(
                f"{wandb.run.dir}/agent.pt",
                make_env,
                args.env_id,
                eval_episodes=3,
                run_name=run_name,
                model=Agent,
                device="cpu",
                capture_video=True,
                args=args,
            )

            if os.path.exists(f"videos/{run_name}"):
                for video_file in os.listdir(f"videos/{run_name}"):
                    if video_file.endswith(".mp4"):
                        wandb.log({
                            f"videos/eval_{video_file}": wandb.Video(f"videos/{run_name}/{video_file}")
                        }, step=global_step)
            for i in range(len(episodic_returns)):  
                writer.add_scalar(f"eval/episodic_return", episodic_returns[i], global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()