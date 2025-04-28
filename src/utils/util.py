import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from steakhouse_ai_py.mdp.steakhouse_mdp import SteakhouseGridworld
from steakhouse_ai_py.mdp.steakhouse_env import CommsSteakhouseEnv
from steakhouse_ai_py.planners.steak_planner import SteakMediumLevelActionManager
from configs.args import Args

def make_env(env_id, idx, capture_video, run_name, args=None):
    def thunk():
        if capture_video and idx == 0:
            # Use "rgb_array" render mode for headless environments
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            # For non-recording environments, use "rgb_array" to avoid display issues
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def make_steakhouse_env(args: Args):
    layout_name = args.layout_name
    world_mdp = SteakhouseGridworld.from_layout_name(layout_name)
    mlam = SteakMediumLevelActionManager.from_pickle_or_compute(
            world_mdp,
            {
                'start_orientations': True,
                'wait_allowed': True,
                'counter_goals': [],
                'counter_drop': world_mdp.terrain_pos_dict['X'],
                'counter_pickup': world_mdp.terrain_pos_dict['X'],
                'same_motion_goals': True,
                "enable_same_cell": True,
            },
            custom_filename=None,
            force_compute=False,
            info=False,
        )
    # Pass render_mode="rgb_array" to ensure headless rendering
    env = CommsSteakhouseEnv.from_mdp(world_mdp, horizon=args.max_episode_steps, mlam=mlam, discretization=args.discretization, render_mode="rgb_array")
    # env.reset(rand_start=args.rand_start)
    return env
