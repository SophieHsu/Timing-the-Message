import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base import BaseAgent
from utils.util import layer_init
from feature_extractors.highway_features import HighwayFeaturesExtractor

class MLPAgent(BaseAgent):
    def __init__(self, envs, args):
        super().__init__(envs, args)
        input_dim = np.array(envs.single_observation_space.shape).prod()
        self.feature_extractor = args.feature_extractor

        if args.feature_extractor == "highway":
            attention_network_kwargs = dict(
                in_size=5 * 15,
                embedding_layer_kwargs={"in_size": 7, "layer_sizes": [64, 64], "reshape": False},
                attention_layer_kwargs={"feature_size": 64, "heads": 2},
            )
            self.feature_extract = HighwayFeaturesExtractor(envs.single_observation_space, **attention_network_kwargs)
            input_dim = self.feature_extract.features_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space[-1].n), std=0.01),
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
    def __init__(self, envs, args):
        super().__init__(envs, args)
        input_dim = np.array(envs.single_observation_space.shape).prod()
        hidden_dim = 64
        self.condition_dim, self.id_dim, self.length_dim = envs.single_action_space.shape

        self.notifier = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        
        self.condition_head = nn.Linear(hidden_dim, self.condition_dim)
        self.id_head = nn.Linear(hidden_dim, self.id_dim)
        self.length_head = nn.Linear(hidden_dim, self.length_dim)

    def get_action_and_value(self, x, action=None):
        if self.feature_extractor == "highway":
            x = self.feature_extract(x)
        features = self.notifier(x)
        condition_logits = self.condition_head(features)
        id_logits = self.id_head(features)
        length_logits = self.length_head(features)

        condition_probs = Categorical(logits=condition_logits)
        id_probs = Categorical(logits=id_logits)
        length_probs = Categorical(logits=length_logits)

        if action is None:
            condition = condition_probs.sample()
            id = id_probs.sample()
            length = length_probs.sample()
            action = (condition, id, length)

        logprob = condition_probs.log_prob(action[0]) + id_probs.log_prob(action[1]) + length_probs.log_prob(action[2])
        entropy = condition_probs.entropy() + id_probs.entropy() + length_probs.entropy()

        return action, logprob, entropy, self.critic(x)
    
