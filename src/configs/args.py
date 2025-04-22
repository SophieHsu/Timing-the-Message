from dataclasses import dataclass
import copy

@dataclass
class Args:
    # Experiment settings
    exp_name: str = "ppo"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = None
    cuda: bool = True
    cuda_device: str = "cuda"
    track: bool = True
    wandb_project_name: str = "timing"
    wandb_entity: str = "tri"
    capture_video: bool = False
    save_freq: int = 1
    model_path: str = None  # Path to the trained agent model
    model_run_id: str = "4ccwlxqs" # 03cwsek5
    # Environment settings
    env_id: str = "multi-merge-v0" # "DangerZoneLunarLander" #"multi-merge-v0" "LargeRewardNotiLunarLander"
    total_timesteps: int = int(1e9)
    highway_features_dim: int = 64
    max_episode_steps: int = 1000
    # Agent settings
    agent_type: str = "mlp"  # Options: "mlp", "lstm", "transformer", "heuristic"
    trainer_type: str = "base"  # Options: "base", "lstm", "transformer", "heuristic", "blocking"
    use_condition_head: bool = True
    agent_obs_mode: str = "history" # Options: "history"
    rollout_reward_buffer_steps: int = 5

    # Human agent settings
    human_agent_type: str = "IDM" # Options: "None", "mlp", "lstm", "transformer", "IDM"
    human_agent_run_id: str = "xlq34dpt"
    human_agent_path: str = None
    human_utterance_memory_length: int = 10
    human_reaction_delay: int = 0
    human_comprehend_bool: bool = False
    fix_overwrite: bool = False

    # PPO settings
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    
    # Transformer specific settings
    context_len: int = 10
    n_blocks: int = 4
    h_dim: int = 32
    n_heads: int = 4
    
    # LSTM specific settings
    lstm_size: int = 32
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    
    # MLP specific settings
    mlp_hidden_dims: list = None

    # Feature extractor settings
    feature_extractor: str = "highway" # Options: "highway", "none"
    num_vehicles: int = 8
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 64] 