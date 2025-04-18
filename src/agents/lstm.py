import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from utils.util import layer_init
import numpy as np

class LSTMAgent(nn.Module):
    def __init__(self, envs, args, single_observation_space=None):
        super().__init__()
        self.lstm_size = args.lstm_size
        self.single_observation_space = envs.single_observation_space.shape[0] if single_observation_space is None else single_observation_space
        self.network = nn.Sequential(
            layer_init(nn.Linear(self.single_observation_space, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, self.lstm_size)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.lstm_size, 128)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)
        self.actor = layer_init(nn.Linear(128, envs.single_action_space[-1].n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

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
    def __init__(self, envs, args):
        self.human_utterance_memory_length = args.human_utterance_memory_length
        self.agent_obs_mode = args.agent_obs_mode
        if self.agent_obs_mode == "history":
            self.single_observation_space = (np.array(envs.single_observation_space.shape).prod() + (envs.single_action_space.shape[0]-1))*self.human_utterance_memory_length
        else:
            self.single_observation_space = np.array(envs.single_observation_space.shape).prod()
        super().__init__(envs, args, self.single_observation_space)

        self.use_condition_head = args.use_condition_head
        hidden_dim = 64
        self.condition_dim, self.id_dim, self.length_dim, _ = envs.single_action_space.nvec

        if self.use_condition_head:
            self.condition_head = layer_init(nn.Linear(128, self.condition_dim), std=0.01)
        self.id_head = layer_init(nn.Linear(128, self.id_dim), std=0.01)
        self.length_head = layer_init(nn.Linear(128, self.length_dim), std=0.01)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)

        id_logits = self.id_head(hidden)
        length_logits = self.length_head(hidden)
        id_probs = Categorical(logits=id_logits)
        length_probs = Categorical(logits=length_logits)
        
        
        if self.use_condition_head:
            condition_logits = self.condition_head(hidden)
            condition_probs = Categorical(logits=condition_logits)

        if action is None:
            if self.use_condition_head:
                condition = condition_probs.sample()
            else:
                condition = torch.zeros(self.args.num_envs).to(self.args.device)
            id = id_probs.sample()
            length = length_probs.sample()
            action = torch.stack([condition, id, length], dim=1)

        if self.use_condition_head:
            logprob = condition_probs.log_prob(action[:, 0]) + id_probs.log_prob(action[:, 1]) + length_probs.log_prob(action[:, 2])
            entropy = condition_probs.entropy() + id_probs.entropy() + length_probs.entropy()
        else:
            logprob = id_probs.log_prob(action[:, 1]) + length_probs.log_prob(action[:, 2])
            entropy = id_probs.entropy() + length_probs.entropy()

        return action, logprob, entropy, self.critic(hidden), lstm_state