from dataclasses import dataclass
import copy

@dataclass
class Args:
    # Experiment settings
    exp_name: str = "ppo"
    seed: int = 1
    torch_deterministic: bool = True
    device: str = None
    cuda: bool = False
    cuda_device: str = "cuda:1"
    track: bool = True
    wandb_project_name: str = "timing"
    wandb_entity: str = "tri"
    capture_video: bool = False
    save_freq: int = 1
    model_path: str = None  # Path to the trained agent model
    model_run_id: str = None #"iongjjwi" #"sgeo696n" # 03cwsek5
    ray_debug_mode: bool = True  # Run in debug mode with simplified data and CPU
    debug: bool = False
    num_cpus: int = 8
    num_gpus: int = 1
    render: bool = False  # Whether to render locally (True) or in headless mode (False)
    # Environment settings
    headless: bool = True
    env_id: str = "steakhouse" # "DangerZoneLunarLander" #"multi-merge-v0" "LargeRewardNotiLunarLander" "complex-noti-multi-merge-v0" "simple-noti-multi-merge-v0"
    total_timesteps: int = int(1e9)
    highway_features_dim: int = 64
    steakhouse_feature_dim: int = 128
    steakhouse_one_dim_obs_dim: int = 13
    max_episode_steps: int = 65
    layout_random: bool = False
    
    # Agent settings
    agent_type: str = "lstm"  # Options: "mlp", "lstm", "transformer", "heuristic"
    trainer_type: str = "cooking"  # Options: "base", "lstm", "transformer", "heuristic", "blocking", "cooking"
    use_condition_head: bool = True
    noti_action_length: int = None # determined by the environment
    agent_obs_mode: str = "history" # Options: "history"
    rollout_reward_buffer_steps: int = 5
    save_trajectory: bool = False
    random_danger_zone: bool = False
    noti_penalty_weight: float = 1.0
    agent_step_penalty: float = 0.00
    env_reward_mode: bool = False
    one_dim_obs: bool = False
    early_termination: bool = False
    new_noti_penalty: float = 0.00
    block_rollout: bool = False
    overfit: bool = False
    
    # Human agent settings
    human_agent_type: str = "chef" # Options: "None", "mlp", "lstm", "transformer", "IDM", "chef"
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
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 1
    
    # MLP specific settings
    mlp_hidden_dims: list = None

    # Feature extractor settings
    feature_extractor: str = "none" # Options: "highway", "none", "steakhouse"
    num_vehicles: int = 8
    def __post_init__(self):
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [64, 64] 
        # Set feature extractor to "steakhouse" if env_id is "steakhouse"
        if self.env_id == "steakhouse":
            self.feature_extractor = "steakhouse"

    # Steakhouse specific settings
    layout_name: str = "small"
    participant_id: int = 0
    log_file_name: str = None
    total_time: int = None
    fov: int = 120
    record_video: bool = None
    order_list: list = None
    rand_start: bool = False
    discretization: str = "simple"
    auto_unstuck: bool = False
    VISION_LIMIT: bool = True
    VISION_BOUND: int = 120
    VISION_MODE: str = "grid"
    EXPLORE: bool = False
    KB_UPDATE_DELAY: int = 0
    KB_ACKN_PROB: bool = False
    drop_on_counter: bool = False