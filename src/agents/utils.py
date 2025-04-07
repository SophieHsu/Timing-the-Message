import torch
import gymnasium as gym
import numpy as np
from typing import Callable
import os

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    args: dict = {},
):
    # Create videos directory for evaluation
    os.makedirs("videos/eval", exist_ok=True)
    
    # Use RecordVideo for both wandb and non-wandb cases
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, "eval")])
    agent = model(envs, args).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()

    num_envs = 1
    obs = torch.zeros((args.num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, num_envs) + envs.single_action_space[-1].shape).to(device)
    rewards = torch.zeros((args.num_steps, num_envs)).to(device)
    env_steps = torch.zeros((args.num_steps, num_envs)).to(device)
    times_contexts = torch.zeros((args.num_steps, num_envs, args.context_len), dtype=torch.long).to(device)
    obs_contexts = torch.zeros((args.num_steps, num_envs, args.context_len) + envs.single_observation_space.shape).to(device)
    action_contexts = torch.zeros((args.num_steps, num_envs, args.context_len) + envs.single_action_space[-1].shape, dtype=torch.long).to(device)

    next_obs, _ = envs.envs[0].reset()
    next_obs = torch.Tensor(next_obs).to(device)
    env_step = torch.zeros(num_envs).to(device)

    episodic_returns = []
    step = 0
    while len(episodic_returns) < eval_episodes:
        obs[step] = next_obs
        env_steps[step] = env_step

        if step < args.context_len:
            # For each environment, create the context
            times_context = torch.zeros((args.context_len, num_envs), dtype=torch.long, device=device)
            times_context[:step] = env_steps[:step]
            times_context[step:] = step
            
            obs_context = torch.zeros((args.context_len, num_envs) + obs.shape[2:], dtype=torch.float32, device=device)
            obs_context[:step] = obs[:step]
            obs_context[step:] = obs[step].unsqueeze(0).repeat(args.context_len-step, 1, 1)
            
            action_context = torch.zeros((args.context_len, num_envs) + actions.shape[2:], dtype=torch.long, device=device)
            action_context[:step] = actions[:step]
            action_context[step:] = actions[step].unsqueeze(0).repeat(args.context_len-step, 1)

            times_context = times_context.transpose(0, 1)
            obs_context = obs_context.transpose(0, 1)
            action_context = action_context.transpose(0, 1)
        else:
            times_context = env_steps[step-args.context_len:step].transpose(0, 1)
            obs_context = obs[step-args.context_len:step].transpose(0, 1)
            action_context = actions[step-args.context_len:step].transpose(0, 1)

        times_contexts[step] = times_context.long()
        obs_contexts[step] = obs_context
        action_contexts[step] = action_context.long()

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value([times_context.long(), obs_context, action_context.long()])
        actions[step] = action

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.envs[0].step([(0,0,0), action.cpu().numpy().item()])
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor([reward]).to(device).view(-1)
        next_obs, next_done = torch.Tensor(np.array([next_obs])).to(device), torch.Tensor([next_done]).to(device)
        env_step = torch.where(next_done==1.0, torch.zeros_like(env_step), env_step + 1)
        
        if next_done:
            episodic_returns += [rewards[step].sum().item()]

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]

    return episodic_returns