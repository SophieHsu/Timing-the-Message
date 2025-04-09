import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.mlp import MLPAgent
from agents.lstm import LSTMAgent
from agents.transformers import TransformerAgent
import wandb
from configs.args import Args

class HumanAgent:
    def __init__(self, envs, args):
        self.envs = envs
        self.args = args

        api = wandb.Api()
        run = api.run(f"yachuanh/timing/{args.human_agent_run_id}")
        human_agent_path = run.config['filepath'] + "/agent.pt"

        if args.human_agent_type == "mlp":
            self.policy_network = MLPAgent(envs, args)
        elif args.human_agent_type == "lstm":
            self.policy_network = LSTMAgent(envs, args)
        elif args.human_agent_type == "transformer":
            self.policy_network = TransformerAgent(envs, args)
        else:
            raise ValueError(f"Unknown human agent type: {args.human_agent_type}")
        self.policy_network.load_state_dict(torch.load(human_agent_path))
        self.policy_network.eval()

        self.utterance_memory = np.array([(0,0,0)]*self.args.human_utterance_memory_length)
        self.overwrite_action = np.array([None]*self.args.num_envs)

    def process_utterance(self, utterance):
        # Update utterance memory with the new utterance
        self.overwrite_action = np.array([None]*self.args.num_envs)
        self.utterance_memory = np.concatenate([self.utterance_memory[:,1:], utterance.unsqueeze(0)], axis=1)

        # Initialize arrays for vectorized processing
        num_envs = self.utterance_memory.shape[0]
        track_lengths = np.zeros(num_envs, dtype=int)
        track_noti_actions = np.array([None] * num_envs)
        track_noti_action_lengths = np.zeros(num_envs, dtype=int)
        
        # Process all environments at once
        # We'll iterate through the utterance memory in reverse order
        is_done = np.zeros(num_envs, dtype=bool)
        for i in range(self.utterance_memory.shape[1] - 1, -1, -1):
            # Get all utterances at the current position across all environments
            current_utters = self.utterance_memory[:, i]
            
            # Create masks for different utterance types
            is_negative_one = np.array([tuple(utter) == (-1, 0, 0) for utter in current_utters])
            is_zero = np.array([tuple(utter) == (0, 0, 0) for utter in current_utters])
            is_other = ~(is_negative_one | is_zero)
            is_done[is_zero] = True
            
            # Process negative one utterances
            track_lengths[is_negative_one] += 1
            
            # Process zero utterances
            self.overwrite_action[is_zero] = None
            
            # Process other utterances
            for env_idx in np.where(is_other)[0]:
                utter = current_utters[env_idx]
                if track_noti_actions[env_idx] is None:
                    track_lengths[env_idx] = 1
                    _, track_noti_actions[env_idx], track_noti_action_lengths[env_idx] = tuple(utter)
                else:
                    self.overwrite_action[env_idx] = None
            
            # Check if track_length >= track_noti_action_length for any environment
            valid_lengths = (track_noti_action_lengths > 0) & (track_lengths >= track_noti_action_lengths) & (track_noti_actions != None)
            self.overwrite_action[valid_lengths] = track_noti_actions[valid_lengths]
            is_done[valid_lengths] = True
            
            # Break early if all environments have been processed
            if np.all(self.overwrite_action != None) or np.all(is_done):
                break

    def get_action(self, obs, utterance):
        self.policy_network.eval()
        with torch.no_grad():
            action, _, _, _ = self.policy_network.get_action_and_value(obs)

        # Update action based on utterance
        self.process_utterance(utterance)

        # Use the overwrite_action if it's set
        update_action_idx = np.where(self.overwrite_action != None)[0]
        action[update_action_idx] = self.overwrite_action[update_action_idx]
        
        return action
    