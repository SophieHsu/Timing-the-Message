import os
import torch
import wandb
import time
import tyro
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import gymnasium_envs

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.args import Args
from agents.mlp import MLPAgent
from agents.transformers import TransformerAgent
from agents.lstm import LSTMAgent
from utils.training import BaseTrainer, LSTMTrainer, TransformerTrainer
from utils.util import make_env

def main():
    args = tyro.cli(Args)
    # args = Args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id.split('/')[-1]}_{args.exp_name}_{args.learning_rate}_{args.seed}_{int(time.time())}"
    
    # Initialize wandb
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{args.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create vectorized environment
    envs = SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name) for i in range(args.num_envs)]
    )
    
    # Create agent based on args
    if args.agent_type == "mlp":
        agent = MLPAgent(envs, args).to(device)
    elif args.agent_type == "lstm":
        agent = LSTMAgent(envs, args).to(device)
    elif args.agent_type == "transformer":
        agent = TransformerAgent(envs, args).to(device)
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")
    
    # Train agent
    if args.agent_type == "mlp":
        trainer = BaseTrainer(agent, envs, args, writer, run_name, device)
    elif args.agent_type == "lstm":
        trainer = LSTMTrainer(agent, envs, args, writer, run_name, device)
    elif args.agent_type == "transformer":
        trainer = TransformerTrainer(agent, envs, args, writer, run_name, device)
    trainer.train()

if __name__ == "__main__":
    main() 