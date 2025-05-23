"""
this extremely minimal GPT model is based on:
Misha Laskin's tweet: 
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

the above colab has a bug while applying masked_fill which is fixed in the
following code

"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Dropout(drop_p),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm with residual connections
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x
    
class TransformerCritic(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, 
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 2 * context_len # state, action
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim)
        )
        
        # discrete actions
        self.embed_action = nn.Sequential(
            nn.Embedding(act_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim)
        )
        
        ### prediction heads
        self.predict_value = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, timesteps, states, actions):
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        # stack states and actions and reshape sequence as
        # (s1, a1, s2, a2 ...)
        h = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 2 * T, self.h_dim)

        h = self.embed_ln(h)
        
        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 2 x T x h_dim) and
        # h[:, 0, t] is conditioned on s_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on s_0, a_0 ... s_t, a_t
        h = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        value_preds = self.predict_value(h[:,1])  # predict value given s, a
    
        return value_preds[:, -1]


class TransformerPolicy(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, 
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 2 * context_len # state, action
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim)
        )
        
        # discrete actions
        self.embed_action = nn.Sequential(
            nn.Embedding(act_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim)
        )
        
        ### prediction heads
        self.predict_action = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, act_dim)
        )

    def forward(self, timesteps, states, actions):
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        # stack states and actions and reshape sequence as
        # (s1, a1, s2, a2 ...)
        h = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 2 * T, self.h_dim)

        h = self.embed_ln(h)
        
        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 2 x T x h_dim) and
        # h[:, 0, t] is conditioned on s_0, a_0 ... s_t
        # h[:, 1, t] is conditioned on s_0, a_0 ... s_t, a_t
        h = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        action_preds = self.predict_action(h[:,0])  # predict action given s
    
        return action_preds[:, -1]

    def action(self, timesteps, states, actions):
        action_preds = self.forward(timesteps, states, actions)
        return action_preds.argmax(dim=-1)


class TransformerAgent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.critic = TransformerCritic(state_dim=np.array(envs.single_observation_space.shape).prod(), 
                                        act_dim=envs.single_action_space[-1].n, 
                                        context_len=args.context_len, 
                                        n_blocks=args.n_blocks, 
                                        h_dim=args.h_dim, 
                                        n_heads=args.n_heads, 
                                        drop_p=0.1)
        
        self.actor = TransformerPolicy(state_dim=np.array(envs.single_observation_space.shape).prod(), 
                                       act_dim=envs.single_action_space[-1].n, 
                                       n_blocks=args.n_blocks, 
                                       h_dim=args.h_dim, 
                                       context_len=args.context_len, 
                                       n_heads=args.n_heads, 
                                       drop_p=0.1)

    def get_value(self, x):
        if isinstance(x, list):
            times, states, actions = x
            return self.critic(times, states, actions)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if isinstance(x, list):
            times, states, actions = x
            logits = self.actor(times, states, actions)

            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(times, states, actions)
        else:
            logits = self.actor(x)

            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)

