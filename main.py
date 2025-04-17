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
import highway_env

from src.configs.args import Args
from src.utils.util import make_env
from src.utils.evaluate import HeuristicEvaluator

os.environ["OFFSCREEN_RENDERING"] = "1"

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Initialize wandb
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.exp_name,
            monitor_gym=False,
            save_code=True,
        )
        wandb.config.update({"filepath": wandb.run.dir})

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
    if args.device is None:
        args.device = device

    eval_count = 0
    num_eval = 10

    # Create vectorized environment
    args.num_envs = 1

    for i in range(num_eval):
        evaluator = HeuristicEvaluator(args, args.exp_name)
        episodic_returns = evaluator.evaluate(
            make_env=make_env,
            eval_episodes=1,
            device=device,
            capture_video=True,
            visualize=True,
        )

        if os.path.exists(f"videos/{args.exp_name}"):
            for video_file in os.listdir(f"videos/{args.exp_name}"):
                if video_file.endswith(".mp4"):
                    wandb.log({
                        f"videos/eval_{video_file}": wandb.Video(f"videos/{args.exp_name}/{video_file}")
                    }, step=i)

        for j in range(len(episodic_returns)):  
            writer.add_scalar(f"eval/episodic_return", episodic_returns[j], i)
