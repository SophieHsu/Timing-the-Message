import torch
import numpy as np
import os
import sys
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.mlp import MLPAgent
from agents.lstm import LSTMAgent
from agents.transformers import TransformerAgent
import wandb
from steakhouse_ai_py.agents.steak_agent import SteakLimitVisionHumanModel
from overcooked_ai_py.mdp.actions import Action
class HumanAgent:
    def __init__(self, envs, args, device, num_envs=None):
        self.envs = envs
        self.noti_action_length = args.noti_action_length
        self.args = args
        self.device = device    
        
        if args.human_agent_type not in ["IDM", "chef"]:
            api = wandb.Api()
            run = api.run(f"tri/timing/{args.human_agent_run_id}")
            human_agent_path = run.config['filepath'] + "/agent.pt"
            # human_agent_path = "/home/sophie.hsu.pi/Timing-the-Message/wandb/run-20250409_201211-xlq34dpt/files/agent.pt"
        
            if args.human_agent_type == "mlp":
                print(envs.single_observation_space)
                self.policy_network = MLPAgent(args, 8, envs.single_action_space).to(self.device)
            elif args.human_agent_type == "lstm":
                self.policy_network = LSTMAgent(args, envs.single_observation_space).to(self.device)
            elif args.human_agent_type == "transformer":
                self.policy_network = TransformerAgent(envs, args).to(self.device)
            else:
                raise ValueError(f"Unknown human agent type: {args.human_agent_type}")
            self.policy_network.load_state_dict(torch.load(human_agent_path, map_location=self.device, weights_only=True))
            self.policy_network.eval()

        self.fixed_overwrite_length = 3
        if self.args.rand_human_reaction_delay_sigma != 0:
            rand_delay = int(np.random.normal(2, self.args.rand_human_reaction_delay_sigma))
            if rand_delay < 0:
                rand_delay = 0
            if rand_delay > 4:
                rand_delay = 4
            self.reaction_delay = rand_delay
        else:
            self.reaction_delay = self.args.human_reaction_delay

        self.num_envs = num_envs if num_envs is not None else self.args.num_envs
        self.reset()

    def reset(self, idx=None):
        if idx is None:
            self.utterance_memory = np.array([[tuple([0]*self.noti_action_length)]*self.args.human_utterance_memory_length]*self.num_envs)
            self.overwrite_action = np.array([-1]*self.num_envs, dtype=np.int32)
            self.tmp_overwrite_action = np.array([-1]*self.num_envs, dtype=np.int32)
            self.track_overwrite = np.zeros(self.num_envs, dtype=np.int32)
            self.track_reaction_delay = np.zeros(self.num_envs, dtype=np.int32)
            self.overwrite_length = np.array([0]*self.num_envs, dtype=np.int32)
            self.tmp_overwrite_length = np.array([0]*self.num_envs, dtype=np.int32)

        else:
            self.utterance_memory[idx] = np.array([tuple([0]*self.noti_action_length)]*self.args.human_utterance_memory_length)
            self.overwrite_action[idx] = -1
            self.tmp_overwrite_action[idx] = -1
            self.track_overwrite[idx] = 0
            self.track_reaction_delay[idx] = 0
            self.overwrite_length[idx] = 0
            self.tmp_overwrite_length[idx] = 0


    def process_utterance(self, utterance):
        # Update utterance memory with the new utterance
        self.utterance_memory = np.concatenate([self.utterance_memory[:,1:], utterance.reshape(-1, 1, self.noti_action_length)], axis=1)

        # Initialize arrays for vectorized processing
        num_envs = self.utterance_memory.shape[0]
        track_lengths = np.zeros(num_envs, dtype=int)
        track_noti_actions = np.array([None] * num_envs)
        track_noti_action_lengths = np.zeros(num_envs, dtype=int)
        track_noti_action_reaction_lengths = np.zeros(num_envs, dtype=int)
        
        # Process all environments at once
        # We'll iterate through the utterance memory in reverse order
        is_done = np.zeros(num_envs, dtype=bool)
        for i in range(self.utterance_memory.shape[1] - 1, -1, -1):
            # Get all utterances at the current position across all environments
            current_utters = self.utterance_memory[:, i]
            
            # Create masks for different utterance types
            is_continue_one = np.array([tuple(utter) == tuple([1]+[0]*(self.noti_action_length-1)) for utter in current_utters])
            is_zero = np.array([tuple(utter) == tuple([0]*self.noti_action_length) for utter in current_utters])
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
                    tmp_utter = tuple(utter)
                    track_noti_actions[env_idx], track_noti_action_lengths[env_idx] = tmp_utter[1], tmp_utter[2]
                    if self.noti_action_length > 3:
                        track_noti_action_reaction_lengths[env_idx] = tmp_utter[3]
                    else:
                        track_noti_action_reaction_lengths[env_idx] = tmp_utter[2]
                    if not self.args.human_comprehend_bool:
                        # track_noti_action_lengths[env_idx] = 1
                        track_noti_action_reaction_lengths[env_idx] = 1
                    is_done[env_idx] = True
                else:
                    is_done[env_idx] = True

            # update overwrite length
            valid_update_overwrite_lengths = np.where((track_noti_action_reaction_lengths != track_noti_action_lengths) & (track_lengths == track_noti_action_lengths))[0]
            if not self.args.fix_overwrite:
                self.overwrite_length[valid_update_overwrite_lengths] = track_noti_action_lengths[valid_update_overwrite_lengths]
            else:
                self.overwrite_length[valid_update_overwrite_lengths] = self.fixed_overwrite_length

            # update overwrite action
            valid_lengths = np.where((track_noti_action_lengths > 0) & (track_lengths == track_noti_action_reaction_lengths) & (track_noti_actions != None))[0]
            self.tmp_overwrite_action[valid_lengths] = track_noti_actions[valid_lengths]
            if not self.args.fix_overwrite:
                self.tmp_overwrite_length[valid_lengths] = track_noti_action_lengths[valid_lengths]
            else:
                self.tmp_overwrite_length[valid_lengths] = self.fixed_overwrite_length
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
        self.overwrite_length[start_overwrite_idx] = self.tmp_overwrite_length[start_overwrite_idx]
        self.tmp_overwrite_length[start_overwrite_idx] = 0
        self.track_overwrite[start_overwrite_idx] = 1
        self.track_reaction_delay[start_overwrite_idx] = 0

    def get_action(self, obs, utterance):
        with torch.no_grad():
            action, _, _, _ = self.policy_network.get_action_and_value(obs[:, :8])

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
    def __init__(self, envs, args, device, num_envs=None):
        super().__init__(envs, args, device, num_envs)

    
    def process_utterance(self, utterance):
        # Update utterance memory with the new utterance
        self.utterance_memory = np.concatenate([self.utterance_memory[:,1:], utterance.reshape(self.num_envs, 1, self.noti_action_length)], axis=1)

        # Initialize arrays for vectorized processing
        num_envs = self.utterance_memory.shape[0]
        track_lengths = np.zeros(num_envs, dtype=int)
        track_noti_actions = np.array([None] * num_envs)
        track_noti_action_lengths = np.zeros(num_envs, dtype=int)
        track_noti_action_reaction_lengths = np.zeros(num_envs, dtype=int)
        # Process all environments at once
        # We'll iterate through the utterance memory in reverse order
        is_done = np.zeros(num_envs, dtype=bool)
        for i in range(self.utterance_memory.shape[1] - 1, -1, -1):
            # Get all utterances at the current position across all environments
            current_utters = self.utterance_memory[:, i]
            
            # Create masks for different utterance types
            is_continue_one = np.array([tuple(utter) == tuple([1]+[0]*(self.noti_action_length-1)) for utter in current_utters])
            is_zero = np.array([tuple(utter) == tuple([0]*self.noti_action_length) for utter in current_utters])
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
                    tmp_utter = tuple(utter)
                    track_noti_actions[env_idx], track_noti_action_lengths[env_idx] = tmp_utter[1], tmp_utter[2]
                    if self.noti_action_length > 3:
                        track_noti_action_reaction_lengths[env_idx] = tmp_utter[3]
                    else:
                        track_noti_action_reaction_lengths[env_idx] = tmp_utter[2]
                    if not self.args.human_comprehend_bool:
                        # track_noti_action_lengths[env_idx] = 1
                        track_noti_action_reaction_lengths[env_idx] = 1
                    is_done[env_idx] = True
                else:
                    is_done[env_idx] = True

            # update overwrite length
            valid_update_overwrite_lengths = np.where(((track_noti_actions != 0) & (track_noti_actions != 2)) & (track_noti_action_reaction_lengths != track_noti_action_lengths) & (track_lengths == track_noti_action_lengths))[0]
            self.overwrite_length[valid_update_overwrite_lengths] = track_noti_action_lengths[valid_update_overwrite_lengths]

            valid_lengths = np.where((track_noti_action_lengths > 0) & (track_lengths == track_noti_action_reaction_lengths) & (track_noti_actions != None))[0]
            self.tmp_overwrite_action[valid_lengths] = track_noti_actions[valid_lengths]
            if not self.args.fix_overwrite:
                # self.tmp_overwrite_length[valid_lengths] = (track_noti_action_reaction_lengths[valid_lengths]-2)*2 + 2 # 5length = 8overwrite, 2length = 2overwrite
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
        action = np.array([-1]*self.num_envs) 

        # Update action based on utterance
        self.process_utterance(utterance)

        self.update_overwrite_tracking()

        # Use the overwrite_action if it's set
        update_action_idx = np.where((self.overwrite_action != -1) & (self.track_overwrite > 0))[0]
        action[update_action_idx] = self.overwrite_action[update_action_idx]

        current_overwrite_flag = np.array((self.overwrite_action != -1) & (self.track_overwrite > 0), dtype=np.int32)
        
        self.update_reaction_delay_tracking()
        
        return action, current_overwrite_flag

class HumanChefAgent(HumanAgent):
    def __init__(self, envs, args, device, num_envs=1):
        super().__init__(envs, args, device, num_envs)
        self.steakhouse_planner = SteakLimitVisionHumanModel(
            mlam=envs.mlam,
            start_state=envs.state,
            auto_unstuck=False,
            explore=False,
            vision_limit=True,
            vision_bound=120,
            vision_mode="grid",
            kb_update_delay=0,
            kb_ackn_prob=False,
            drop_on_counter=self.args.drop_on_counter,
            debug=False,
        )


    def process_utterance(self, utterance):
        # Update utterance memory with the new utterance
        self.utterance_memory = np.concatenate([self.utterance_memory[:,1:], utterance.reshape(self.num_envs, 1, self.noti_action_length)], axis=1)

        # Initialize arrays for vectorized processing
        num_envs = self.utterance_memory.shape[0]
        track_lengths = np.zeros(num_envs, dtype=int)
        track_noti_actions = np.array([None] * num_envs)
        track_noti_action_lengths = np.zeros(num_envs, dtype=int)
        track_noti_action_reaction_lengths = np.zeros(num_envs, dtype=int)
        # Process all environments at once
        # We'll iterate through the utterance memory in reverse order
        is_done = np.zeros(num_envs, dtype=bool)
        for i in range(self.utterance_memory.shape[1] - 1, -1, -1):
            # Get all utterances at the current position across all environments
            current_utters = self.utterance_memory[:, i]
            
            # Create masks for different utterance types
            is_continue_one = np.array([tuple(utter) == tuple([1]+[0]*(self.noti_action_length-1)) for utter in current_utters])
            is_zero = np.array([tuple(utter) == tuple([0]*self.noti_action_length) for utter in current_utters])
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
                    tmp_utter = tuple(utter)
                    if self.args.discretization == "shortvlong":
                        track_noti_actions[env_idx] = tmp_utter[1]
                        if tmp_utter[1] == 4:
                            track_noti_action_lengths[env_idx] = 5
                        else:
                            track_noti_action_lengths[env_idx] = 2
                    elif self.args.discretization == "short4vlong3":
                        track_noti_actions[env_idx] = tmp_utter[1]
                        if tmp_utter[1] > 3:
                            track_noti_action_lengths[env_idx] = 5
                        else:
                            track_noti_action_lengths[env_idx] = 2
                    else:
                        track_noti_actions[env_idx], track_noti_action_lengths[env_idx] = tmp_utter[1], tmp_utter[2]
                    track_noti_action_reaction_lengths[env_idx] = track_noti_action_lengths[env_idx]
                    if not self.args.human_comprehend_bool:
                        track_noti_action_lengths[env_idx] = 1
                    is_done[env_idx] = True
                else:
                    is_done[env_idx] = True

            if self.args.discretization != "shortvlong":
                # update overwrite action to indicate to update knowledge directly once the full length (>=5) is reached
                if self.args.human_comprehend_bool:
                    valid_update_overwrite_lengths = np.where((track_noti_action_reaction_lengths != track_noti_action_lengths) & (track_lengths == track_noti_action_lengths))[0]
                    self.track_overwrite[valid_update_overwrite_lengths] = 1
                    self.overwrite_length[valid_update_overwrite_lengths] = 2
                    self.overwrite_action[valid_update_overwrite_lengths] = 4 # indicating to update knowledge directly
            else:
                # update overwrite length
                valid_update_overwrite_lengths = np.where((track_noti_action_reaction_lengths != track_noti_action_lengths) & (track_lengths == track_noti_action_lengths))[0]
                self.overwrite_length[valid_update_overwrite_lengths] = track_noti_action_lengths[valid_update_overwrite_lengths]

            valid_lengths = np.where((track_noti_action_lengths > 0) & (track_lengths == track_noti_action_reaction_lengths) & (track_noti_actions != None))[0]
            self.tmp_overwrite_action[valid_lengths] = track_noti_actions[valid_lengths]
            if not self.args.fix_overwrite:
                self.tmp_overwrite_length[valid_lengths] = track_noti_action_reaction_lengths[valid_lengths]

            # can only change lane once
            only_once = np.where((track_noti_actions == 4) & (track_noti_action_lengths > 0) & (track_lengths == track_noti_action_lengths) & (track_noti_actions != None))[0]
            self.tmp_overwrite_length[only_once] = 1

            only_twice = np.where((track_noti_actions != 4) & (track_noti_action_lengths > 0) & (track_lengths == track_noti_action_lengths) & (track_noti_actions != None))[0]
            self.tmp_overwrite_length[only_twice] = 2

            self.track_reaction_delay[valid_lengths] = 1
            is_done[valid_lengths] = True
            
            # Break early if all environments have been processed
            if np.all(is_done):
                break

    def update_obj_id(self, env_state, state_type):
        for category in self.steakhouse_planner.knowledge_base[state_type]:
            for obj_id in self.steakhouse_planner.knowledge_base[state_type][category]:
                if obj_id not in self.steakhouse_planner.knowledge_base and isinstance(obj_id, int):
                    # If the object ID is not in the knowledge base, try to find it in the state
                    for obj in env_state.all_objects_list:
                        if obj.id == obj_id:
                            self.steakhouse_planner.knowledge_base[obj_id] = obj
                            break

    def get_action(self, env_state, utterance):
        action, _ = self.steakhouse_planner.action(env_state)

        # Update action based on utterance
        self.process_utterance(utterance)
        
        self.update_overwrite_tracking()

        # Use the overwrite_action if it's set
        update_action_idx = np.where((self.overwrite_action != -1) & (self.track_overwrite > 0))[0]
        if len(update_action_idx) > 0:
            overwrite_action = self.overwrite_action[update_action_idx][0]
            if overwrite_action < 4:
                action = Action.INDEX_TO_ACTION[overwrite_action]
            else:
                # knowledge: 4 = grill, 5 = chopping board, 6 = sink
                if overwrite_action == 4:
                    self.steakhouse_planner.knowledge_base['grill_states'] = self.steakhouse_planner.mlam.mdp.get_grill_states(env_state, update_knowledge_base=True)
                    self.update_obj_id(env_state, 'grill_states')
                if overwrite_action == 5:
                    self.steakhouse_planner.knowledge_base['chop_states'] = self.steakhouse_planner.mlam.mdp.get_chopping_board_states(env_state, update_knowledge_base=True)
                    self.update_obj_id(env_state, 'chop_states')
                if overwrite_action == 6:
                    self.steakhouse_planner.knowledge_base['sink_states'] = self.steakhouse_planner.mlam.mdp.get_sink_states(env_state, update_knowledge_base=True)
                    self.update_obj_id(env_state, 'sink_states')
                if overwrite_action == 7:
                    self.steakhouse_planner.init_knowledge_base(env_state)

                action, _ = self.steakhouse_planner.action(env_state)#, vision_bound=0)

        current_overwrite_flag = np.array((self.overwrite_action != -1) & (self.track_overwrite > 0), dtype=np.int32)
        if self.overwrite_action[0] >= 4:
            current_overwrite_flag[0] = 2 + (self.overwrite_action[0] - 4)
        
        self.update_reaction_delay_tracking()
        
        return action, current_overwrite_flag[0]
    