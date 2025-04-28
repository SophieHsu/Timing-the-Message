import os
import torch
import wandb
import time
import tyro
from typing import Optional, Union
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import gymnasium_envs
import highway_env

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.args import Args
from src.agents.mlp import MLPAgent, NotifierMLPAgent
from src.agents.lstm import LSTMAgent, NotifierLSTMAgent
from src.agents.transformers import TransformerAgent 
from src.agents.humans import HumanAgent, HumanDriverAgent, HumanChefAgent
from src.agents.heuristic import HeuristicAgent
from src.utils.training import BaseTrainer, LSTMTrainer, TransformerTrainer, HeuristicTrainer, BlockingTrainer, CookingLSTMTrainer
from src.utils.util import make_env, make_steakhouse_env

def set_rendering_mode(headless=True):
    """Set the rendering mode to either headless or local display"""
    if headless:
        os.environ["OFFSCREEN_RENDERING"] = "1"
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        # Try to use software rendering if hardware acceleration is not available
        os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
        os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"
    else:
        # Remove headless rendering settings if they exist
        os.environ.pop("OFFSCREEN_RENDERING", None)
        os.environ.pop("SDL_VIDEODRIVER", None)
        # Keep PYGAME_HIDE_SUPPORT_PROMPT to avoid unnecessary messages
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        os.environ.pop("MESA_GL_VERSION_OVERRIDE", None)
        os.environ.pop("MESA_GLSL_VERSION_OVERRIDE", None)

class AgentFactory:
    """Factory class for creating different types of agents."""
    
    @staticmethod
    def create_agent(args: Args, envs: SyncVectorEnv, device: torch.device) -> Union[MLPAgent, LSTMAgent, TransformerAgent, HeuristicAgent]:
        """Create an agent based on the specified type."""
        if args.agent_type == "mlp":
            return MLPAgent(args, envs.single_observation_space, envs.single_action_space).to(device)
        elif args.agent_type == "lstm":
            return LSTMAgent(args, envs.single_observation_space, envs.single_action_space).to(device)
        elif args.agent_type == "transformer":
            return TransformerAgent(envs, args).to(device)
        elif args.agent_type == "heuristic":
            return HeuristicAgent(envs, args).to(device)
        else:
            raise ValueError(f"Unknown agent type: {args.agent_type}")

    @staticmethod
    def create_notifier_agent(args: Args, envs: SyncVectorEnv, device: torch.device) -> Union[NotifierMLPAgent, NotifierLSTMAgent]:
        """Create a notifier agent based on the specified type."""
        if args.agent_type == "mlp":
            return NotifierMLPAgent(args, envs.single_observation_space, envs.single_action_space, args.noti_action_length).to(device)
        elif args.agent_type == "lstm":
            return NotifierLSTMAgent(args, envs.single_observation_space, envs.single_action_space).to(device)
        else:
            raise ValueError(f"Unknown notifier agent type: {args.agent_type}")

class HumanAgentFactory:
    """Factory class for creating different types of human agents."""
    
    @staticmethod
    def create_human_agent(args: Args, envs: SyncVectorEnv, device: torch.device) -> Union[HumanAgent, HumanDriverAgent]:
        """Create a human agent based on the specified type."""
        if args.human_agent_type == "IDM":
            return HumanDriverAgent(envs, args, device)
        elif args.human_agent_type == "chef":
            return HumanChefAgent(envs, args, device)
        else:
            return HumanAgent(envs, args, device)

class TrainerFactory:
    """Factory class for creating different types of trainers."""
    
    @staticmethod
    def create_trainer(agent: Union[MLPAgent, LSTMAgent, TransformerAgent, HeuristicAgent], 
                      envs: SyncVectorEnv, args: Args, writer: SummaryWriter, 
                      run_name: str, device: torch.device, human_agent: Optional[HumanAgent] = None):
        """Create a trainer based on the specified type."""
        if args.trainer_type == "base":
            return BaseTrainer(agent, envs, args, writer, run_name, device, human_agent)
        elif args.trainer_type == "lstm":
            return LSTMTrainer(agent, envs, args, writer, run_name, device, human_agent)
        elif args.trainer_type == "transformer":
            return TransformerTrainer(agent, envs, args, writer, run_name, device, human_agent)
        elif args.trainer_type == "heuristic":
            return HeuristicTrainer(agent, envs, args, writer, run_name, device, human_agent)
        elif args.trainer_type == "blocking":
            return BlockingTrainer(agent, envs, args, writer, run_name, device, human_agent)
        elif args.trainer_type == "cooking":
            return CookingLSTMTrainer(agent, envs, args, writer, run_name, device, human_agent, ray_debug_mode=args.ray_debug_mode)
        else:
            raise ValueError(f"Unknown trainer type: {args.trainer_type}")

def setup_wandb(args: Args, run_name: str) -> None:
    """Initialize Weights & Biases logging."""
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
        wandb.config.update({"filepath": wandb.run.dir})

def setup_tensorboard(args: Args) -> SummaryWriter:
    """Initialize TensorBoard logging."""
    writer = SummaryWriter(f"runs/{args.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    return writer

def setup_environment(args: Args) -> SyncVectorEnv:
    """Create and setup the training environment."""
    if args.env_id != "steakhouse":
        envs = SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, args.exp_name) for i in range(args.num_envs)]
        )
        envs.reset()
    else:
        envs = make_steakhouse_env(args)
    return envs

def main():
    # Parse arguments
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id.split('/')[-1]}_{args.exp_name}_{args.learning_rate}_{args.seed}_{int(time.time())}"
    
    # Set rendering mode based on whether we want local display or headless
    set_rendering_mode(headless=not args.render)  # Use args.render to determine mode
    
    # Setup logging
    setup_wandb(args, run_name)
    writer = setup_tensorboard(args)
    
    # Set random seed and device
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.device is None:
        args.device = device
    
    # Setup environment
    envs = setup_environment(args)

    # Update args from envs
    if args.env_id != "steakhouse":
        args.noti_action_length = envs.envs[0].unwrapped.noti_action_length
    else:
        args.noti_action_length = envs.noti_action_length

    # Create human agent if specified
    human_agent = None
    if args.human_agent_type is not None and args.human_agent_type != "None":
        human_agent = HumanAgentFactory.create_human_agent(args, envs, device)
        agent = AgentFactory.create_notifier_agent(args, envs, device)
    else:
        agent = AgentFactory.create_agent(args, envs, device)
    
    # Create trainer and start training
    trainer = TrainerFactory.create_trainer(agent, envs, args, writer, run_name, device, human_agent)
    trainer.train()

if __name__ == "__main__":
    main() 