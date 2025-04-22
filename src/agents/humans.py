import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.mlp import MLPAgent
from agents.lstm import LSTMAgent
from agents.transformers import TransformerAgent
import wandb

class HumanAgent:
    def __init__(self, envs, args, device):
        self.envs = envs
        self.args = args
        self.device = device    
        
        if args.human_agent_type != "IDM":
            # api = wandb.Api()
            # run = api.run(f"yachuanh/timing/{args.human_agent_run_id}")
            # human_agent_path = run.config['filepath'] + "/agent.pt"
            human_agent_path = "/home/sophie.hsu.pi/Timing-the-Message/wandb/run-20250409_201211-xlq34dpt/files/agent.pt"
        
            if args.human_agent_type == "mlp":
                self.policy_network = MLPAgent(envs, args).to(self.device)
            elif args.human_agent_type == "lstm":
                self.policy_network = LSTMAgent(envs, args).to(self.device)
            elif args.human_agent_type == "transformer":
                self.policy_network = TransformerAgent(envs, args).to(self.device)
            else:
                raise ValueError(f"Unknown human agent type: {args.human_agent_type}")
            self.policy_network.load_state_dict(torch.load(human_agent_path, map_location=self.device, weights_only=True))
            self.policy_network.eval()

        self.overwrite_length = 3
        self.reaction_delay = self.args.human_reaction_delay
        self.reset()

    def reset(self):
        self.utterance_memory = np.array([[tuple([0,0,0])]*self.args.human_utterance_memory_length]*self.args.num_envs)
        self.overwrite_action = np.array([-1]*self.args.num_envs, dtype=np.int32)
        self.tmp_overwrite_action = np.array([-1]*self.args.num_envs, dtype=np.int32)
        self.track_overwrite = np.zeros(self.args.num_envs, dtype=np.int32)
        self.track_reaction_delay = np.zeros(self.args.num_envs, dtype=np.int32)
        if not self.args.fix_overwrite:
            self.overwrite_length = np.array([0]*self.args.num_envs, dtype=np.int32)
            self.tmp_overwrite_length = np.array([0]*self.args.num_envs, dtype=np.int32)

    def process_utterance(self, utterance):
        # Update utterance memory with the new utterance
        self.utterance_memory = np.concatenate([self.utterance_memory[:,1:], utterance.reshape(self.args.num_envs, 1, 3)], axis=1)

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
            is_continue_one = np.array([tuple(utter) == (1, 0, 0) for utter in current_utters])
            is_zero = np.array([tuple(utter) == (0, 0, 0) for utter in current_utters])
            is_other = ~(is_continue_one | is_zero) & ~is_done
            is_track = is_continue_one & ~is_done
            is_done[is_zero] = True
            
            # Process negative one utterances
            track_lengths[is_track] += 1
            
            # Process other utterances
            for env_idx in np.where(is_other)[0]:
                utter = current_utters[env_idx]
                if track_noti_actions[env_idx] is None:
                    track_lengths[env_idx] += 1
                    _, track_noti_actions[env_idx], track_noti_action_lengths[env_idx] = tuple(utter)
                    if not self.args.human_comprehend_bool:
                        track_noti_action_lengths[env_idx] = 1
                    is_done[env_idx] = True
                else:
                    is_done[env_idx] = True
            
            # Check if track_length >= track_noti_action_length for any environment
            valid_lengths = np.where((track_noti_action_lengths > 0) & (track_lengths == track_noti_action_lengths) & (track_noti_actions != None))[0]
            self.tmp_overwrite_action[valid_lengths] = track_noti_actions[valid_lengths]
            if not self.args.fix_overwrite:
                self.tmp_overwrite_length[valid_lengths] = track_noti_action_lengths[valid_lengths]
            self.track_reaction_delay[valid_lengths] = 1
            is_done[valid_lengths] = True
            
            # Break early if all environments have been processed
            if np.all(is_done):
                break

    def update_reaction_delay_tracking(self):
        continue_reaction_delay_idx = np.where((self.tmp_overwrite_action != -1) & (self.track_reaction_delay > 0))[0]
        self.track_reaction_delay[continue_reaction_delay_idx] += 1

    def update_overwrite_tracking(self):
        overwrite_complete_idx = np.where((self.overwrite_action != -1) & (self.track_overwrite >= self.overwrite_length))[0]
        self.overwrite_action[overwrite_complete_idx] = -1
        self.track_overwrite[overwrite_complete_idx] = 0

        continue_overwrite_idx = np.where((self.overwrite_action != -1) & (self.track_overwrite > 0))[0]
        self.track_overwrite[continue_overwrite_idx] += 1

        start_overwrite_idx = np.where((self.tmp_overwrite_action != -1) & (self.track_reaction_delay >= self.reaction_delay))[0]
        self.overwrite_action[start_overwrite_idx] = self.tmp_overwrite_action[start_overwrite_idx]
        self.tmp_overwrite_action[start_overwrite_idx] = -1
        if not self.args.fix_overwrite:
            self.overwrite_length[start_overwrite_idx] = self.tmp_overwrite_length[start_overwrite_idx]
            self.tmp_overwrite_length[start_overwrite_idx] = 0
        self.track_overwrite[start_overwrite_idx] = 1
        self.track_reaction_delay[start_overwrite_idx] = 0

    def get_action(self, obs, utterance):
        with torch.no_grad():
            action, _, _, _ = self.policy_network.get_action_and_value(obs)

        # Update action based on utterance
        self.process_utterance(utterance)
        
        self.update_overwrite_tracking()

        # Use the overwrite_action if it's set
        update_action_idx = np.where((self.overwrite_action != -1) & (self.track_overwrite > 0))[0]
        action[update_action_idx] = torch.tensor(self.overwrite_action[update_action_idx], dtype=torch.long).to(self.device)

        current_overwrite_flag = np.array((self.overwrite_action != -1) & (self.track_overwrite > 0), dtype=np.int32)
        
        self.update_reaction_delay_tracking()
        

        return action.cpu().numpy(), current_overwrite_flag
    

class HumanDriverAgent(HumanAgent):
    def __init__(self, envs, args, device):
        super().__init__(envs, args, device)

    
    def process_utterance(self, utterance):
        # Update utterance memory with the new utterance
        self.utterance_memory = np.concatenate([self.utterance_memory[:,1:], utterance.reshape(self.args.num_envs, 1, 3)], axis=1)

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
            is_continue_one = np.array([tuple(utter) == (1, 0, 0) for utter in current_utters])
            is_zero = np.array([tuple(utter) == (0, 0, 0) for utter in current_utters])
            is_other = ~(is_continue_one | is_zero) & ~is_done
            is_track = is_continue_one & ~is_done
            is_done[is_zero] = True
            
            # Process negative one utterances
            track_lengths[is_track] += 1
            
            # Process other utterances
            for env_idx in np.where(is_other)[0]:
                utter = current_utters[env_idx]
                if track_noti_actions[env_idx] is None:
                    track_lengths[env_idx] += 1
                    _, track_noti_actions[env_idx], track_noti_action_lengths[env_idx] = tuple(utter)
                    if not self.args.human_comprehend_bool:
                        track_noti_action_lengths[env_idx] = 1
                    is_done[env_idx] = True
                else:
                    is_done[env_idx] = True
            
            # Check if track_length >= track_noti_action_length for any environment
            valid_lengths = np.where((track_noti_action_lengths > 0) & (track_lengths == track_noti_action_lengths) & (track_noti_actions != None))[0]
            self.tmp_overwrite_action[valid_lengths] = track_noti_actions[valid_lengths]
            if not self.args.fix_overwrite:
                self.tmp_overwrite_length[valid_lengths] = track_noti_action_lengths[valid_lengths]
            
            # can only change lane once
            only_once = np.where(((track_noti_actions == 0) | (track_noti_actions == 2)) & (track_noti_action_lengths > 0) & (track_lengths == track_noti_action_lengths) & (track_noti_actions != None))[0]
            self.tmp_overwrite_length[only_once] = 1

            self.track_reaction_delay[valid_lengths] = 1
            is_done[valid_lengths] = True
            
            # Break early if all environments have been processed
            if np.all(is_done):
                break

    def get_action(self, obs, utterance):
        # ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
        action = np.array([-1]*self.args.num_envs) 

        # Update action based on utterance
        self.process_utterance(utterance)

        self.update_overwrite_tracking()

        # Use the overwrite_action if it's set
        update_action_idx = np.where((self.overwrite_action != -1) & (self.track_overwrite > 0))[0]
        action[update_action_idx] = self.overwrite_action[update_action_idx]

        current_overwrite_flag = np.array((self.overwrite_action != -1) & (self.track_overwrite > 0), dtype=np.int32)
        
        self.update_reaction_delay_tracking()
        
        return action, current_overwrite_flag
