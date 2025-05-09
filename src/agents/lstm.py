import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from src.feature_extractors.steakhouse_features import SteakhouseNotifierFeaturesExtractor


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LSTMAgent(nn.Module):
    def __init__(self, args, single_observation_space=None, single_action_space=None):
        super().__init__()
        self.lstm_size = args.lstm_size
        self.lstm_hidden_dim = args.lstm_hidden_dim
        single_observation_space = single_observation_space.shape[0] if single_observation_space is None else single_observation_space
        self.network = nn.Sequential(
            layer_init(nn.Linear(single_observation_space, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.lstm_size)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.lstm_size, self.lstm_hidden_dim)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)
        self.actor = layer_init(nn.Linear(self.lstm_hidden_dim, single_action_space[-1].n), std=0.01)
        self.critic = layer_init(nn.Linear(self.lstm_hidden_dim, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

    def forward(self, x, lstm_state, done):
        return self.get_action_and_value(x, lstm_state, done)[0]
    

class NotifierLSTMAgent(LSTMAgent):
    def __init__(self, args, single_observation_space=None, single_action_space=None):
        self.args = args
        self.human_utterance_memory_length = args.human_utterance_memory_length
        self.agent_obs_mode = args.agent_obs_mode
        flatten_observation_space = np.array(single_observation_space).prod() if isinstance(single_observation_space, tuple) else np.array(single_observation_space.shape).prod()
        
        if self.args.env_id == "steakhouse" and self.args.one_dim_obs:
            flatten_observation_space = 13 

        if self.agent_obs_mode == "history":
            self.single_observation_space = (flatten_observation_space + (single_action_space.shape[0]-1))*self.human_utterance_memory_length
        else:
            self.single_observation_space = flatten_observation_space + (single_action_space.shape[0]-1)

        input_dim = self.single_observation_space

        if self.args.env_id == "steakhouse" and self.agent_obs_mode == "history" and not self.args.one_dim_obs:
            input_dim = (args.steakhouse_feature_dim + (single_action_space.shape[0]-1))*self.human_utterance_memory_length
        elif self.args.env_id == "steakhouse" and not self.args.one_dim_obs:
            input_dim = (args.steakhouse_feature_dim + (single_action_space.shape[0]-1))

        super().__init__(args, input_dim, single_action_space)
        self.use_react_head = False

        self.use_condition_head = args.use_condition_head
        if len(single_action_space.nvec) == 4:
            self.condition_dim, self.id_dim, self.length_dim, _ = single_action_space.nvec
        else:
            self.use_react_head = True
            self.condition_dim, self.id_dim, self.react_dim, self.length_dim, _ = single_action_space.nvec

        if self.use_condition_head:
            self.condition_head = layer_init(nn.Linear(128, self.condition_dim), std=0.01)
        self.id_head = layer_init(nn.Linear(128, self.id_dim), std=0.01)
        self.length_head = layer_init(nn.Linear(128, self.length_dim), std=0.01)
        if self.use_react_head:
            self.react_head = nn.Linear(128, self.react_dim)
            
        # Initialize feature extractor for steakhouse environment
        if args.env_id == "steakhouse" and not args.one_dim_obs:
            attention_network_kwargs = dict(
                in_size=7*8*23,
                embedding_layer_kwargs={"in_size": 7*8, "layer_sizes": [128, 128], "reshape": False},
                attention_layer_kwargs={"feature_size": 128, "heads": 2},
            )
            self.feature_extractor = SteakhouseNotifierFeaturesExtractor(
                args, 
                single_observation_space, 
                features_dim=args.steakhouse_feature_dim,
                **attention_network_kwargs
            )
    
    def get_value(self, x, lstm_state, done):
        if self.args.env_id == "steakhouse" and not self.args.one_dim_obs:
            # Process through steakhouse feature extractor
            x = self.feature_extractor(x)
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)


    def get_action_and_value(self, x, lstm_state, done, action=None):
        if self.args.env_id == "steakhouse" and not self.args.one_dim_obs:
            # Process through steakhouse feature extractor
            x = self.feature_extractor(x)
        
        hidden, lstm_state = self.get_states(x, lstm_state, done)

        id_logits = self.id_head(hidden)
        length_logits = self.length_head(hidden)
        id_probs = Categorical(logits=id_logits)
        length_probs = Categorical(logits=length_logits)
        
        
        if self.use_condition_head:
            condition_logits = self.condition_head(hidden)
            condition_probs = Categorical(logits=condition_logits)

        if self.use_react_head:
            react_logits = self.react_head(hidden)
            react_probs = Categorical(logits=react_logits)

        if action is None:
            if self.use_condition_head:
                condition = condition_probs.sample()
            else:
                condition = torch.zeros(self.args.num_envs).to(self.args.device)

            if self.use_react_head:
                react = react_probs.sample()
            else:
                react = torch.zeros(self.args.num_envs).to(self.args.device)

            id = id_probs.sample()
            length = length_probs.sample()

            if self.use_react_head:
                action = torch.stack([condition, id, length, react], dim=1)
            else:
                action = torch.stack([condition, id, length], dim=1)

        if self.use_condition_head:
            logprob = condition_probs.log_prob(action[:, 0]) + id_probs.log_prob(action[:, 1]) + length_probs.log_prob(action[:, 2])
            entropy = condition_probs.entropy() + id_probs.entropy() + length_probs.entropy()
        else:
            logprob = id_probs.log_prob(action[:, 1]) + length_probs.log_prob(action[:, 2])
            entropy = id_probs.entropy() + length_probs.entropy()

        if self.use_react_head:
            logprob = logprob + react_probs.log_prob(action[:, 3])
            entropy = entropy + react_probs.entropy()

        return action, logprob, entropy, self.critic(hidden), lstm_state