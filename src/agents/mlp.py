import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from .base import BaseAgent
from src.feature_extractors.highway_features import HighwayNotifierFeaturesExtractor
from src.feature_extractors.steakhouse_features import SteakhouseNotifierFeaturesExtractor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLPAgent(BaseAgent):
    def __init__(self, args, single_observation_space=None, single_action_space=None):
        super().__init__(args)
        self.feature_extractor = args.feature_extractor

        if single_observation_space is None or not isinstance(single_observation_space, np.int64):
            single_observation_space = np.array(single_observation_space.shape).prod()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(single_observation_space, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(single_observation_space, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, single_action_space[-1].n), std=0.01),
        )

    def get_value(self, x):
        if self.feature_extractor == "highway":
            x = self.feature_extract(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.feature_extractor == "highway":
            x = self.feature_extract(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def forward(self, x):
        if self.feature_extractor == "highway":
            x = self.feature_extract(x)
        return self.get_action_and_value(x) 

class NotifierMLPAgent(MLPAgent):
    def __init__(self, args, single_observation_space=None, single_action_space=None, noti_action_length=None):
        self.use_condition_head = args.use_condition_head
        self.use_react_head = False
        self.human_utterance_memory_length = args.human_utterance_memory_length
        self.noti_action_length = noti_action_length
        args.noti_action_length = self.noti_action_length
        flatten_observation_space = np.array(single_observation_space).prod() if isinstance(single_observation_space, tuple) else np.array(single_observation_space.shape).prod()

        self.agent_obs_mode = args.agent_obs_mode
        if self.agent_obs_mode == "history":
            self.single_observation_space = (flatten_observation_space + self.noti_action_length)*self.human_utterance_memory_length
        else:
            self.single_observation_space = flatten_observation_space + self.noti_action_length

        if args.feature_extractor == "highway":
            input_dim = (args.highway_features_dim + (single_action_space.shape[0]-1))*self.human_utterance_memory_length
        elif args.env_id == "steakhouse":
            input_dim = (args.steakhouse_feature_dim + (single_action_space.shape[0]-1))*self.human_utterance_memory_lengthace
        else:
            input_dim = self.single_observation_space

        # Initialize parent class first
        super().__init__(args, input_dim, single_action_space)

        # Initialize feature extractor after parent class
        if args.feature_extractor == "highway" and self.agent_obs_mode == "history":
            attention_network_kwargs = dict(
                in_size=5 * 15,
                embedding_layer_kwargs={"in_size": 7, "layer_sizes": [64, 64], "reshape": False},
                attention_layer_kwargs={"feature_size": 64, "heads": 2},
            )
            self.feature_extract = HighwayNotifierFeaturesExtractor(args, np.array(single_observation_space.shape).prod(), **attention_network_kwargs)
        elif args.env_id == "steakhouse":
            # Initialize steakhouse feature extractor
            attention_network_kwargs = dict(
                in_size=7*8*23,
                embedding_layer_kwargs={"in_size": 23, "layer_sizes": [128, 128], "reshape": False},
                attention_layer_kwargs={"feature_size": 128, "heads": 2},
            )
            self.feature_extract = SteakhouseNotifierFeaturesExtractor(
                args, 
                np.array(single_observation_space.shape).prod(), 
                features_dim=128
            )

        hidden_dim = 64
        if len(single_action_space.nvec) == 4:
            self.condition_dim, self.id_dim, self.length_dim, _ = single_action_space.nvec
        else:
            self.use_react_head = True
            self.condition_dim, self.id_dim, self.react_dim, self.length_dim, _ = single_action_space.nvec

        self.notifier = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        
        if self.use_condition_head:
            self.condition_head = nn.Linear(hidden_dim, self.condition_dim)
        self.id_head = nn.Linear(hidden_dim, self.id_dim)
        self.length_head = nn.Linear(hidden_dim, self.length_dim)
        if self.use_react_head:
            self.react_head = nn.Linear(hidden_dim, self.react_dim)

    def get_value(self, x):
        if self.feature_extractor == "highway" or self.args.env_id == "steakhouse":
            x = self.feature_extract(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.feature_extractor == "highway" or self.args.env_id == "steakhouse":
            x = self.feature_extract(x)
        features = self.notifier(x)

        if self.use_condition_head:
            condition_logits = self.condition_head(features)
            condition_probs = Categorical(logits=condition_logits)

        if self.use_react_head:
            react_logits = self.react_head(features)
            react_probs = Categorical(logits=react_logits)

        id_logits = self.id_head(features)
        length_logits = self.length_head(features)

        id_probs = Categorical(logits=id_logits)
        length_probs = Categorical(logits=length_logits)

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

        return action, logprob, entropy, self.critic(x)
    