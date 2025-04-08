import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.transformers import TransformerPolicy, TransformerCritic

def test_transformer_policy():
    # Test parameters
    state_dim = 4  # Simple state dimension
    act_dim = 2    # Simple action dimension
    n_blocks = 2
    h_dim = 8
    context_len = 3
    n_heads = 2
    batch_size = 4

    # Create model
    model = TransformerPolicy(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=0.1
    )

    # Create dummy inputs
    timesteps = torch.randint(0, 100, (batch_size, context_len))
    states = torch.randn(batch_size, context_len, state_dim)
    actions = torch.randint(0, act_dim, (batch_size, context_len))

    # Test forward pass
    output = model(timesteps, states, actions)
    
    # Check output shape
    assert output.shape == (batch_size, act_dim), f"Expected shape {(batch_size, act_dim)}, got {output.shape}"
    
    # Test action method
    action = model.action(timesteps, states, actions)
    assert action.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {action.shape}"
    assert torch.all(action >= 0) and torch.all(action < act_dim), "Actions should be within valid range"

    print("TransformerPolicy tests passed!")

def test_transformer_critic():
    # Test parameters
    state_dim = 4  # Simple state dimension
    act_dim = 2    # Simple action dimension
    n_blocks = 2
    h_dim = 8
    context_len = 3
    n_heads = 2
    batch_size = 4

    # Create model
    model = TransformerCritic(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=0.1
    )

    # Create dummy inputs
    timesteps = torch.randint(0, 100, (batch_size, context_len))
    states = torch.randn(batch_size, context_len, state_dim)
    actions = torch.randint(0, act_dim, (batch_size, context_len))

    # Test forward pass
    output = model(timesteps, states, actions)
    
    # Check output shape
    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"
    
    print("TransformerCritic tests passed!")

def test_with_simple_env():
    # Create a simple environment with fixed states and actions
    state_dim = 2
    act_dim = 2
    n_blocks = 2
    h_dim = 4
    context_len = 2
    n_heads = 1
    batch_size = 2

    # Create models
    policy = TransformerPolicy(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=0.1
    )

    critic = TransformerCritic(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=0.1
    )

    # Create a simple sequence of states and actions
    timesteps = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    states = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0]],  # First sequence
        [[1.0, 1.0], [2.0, 2.0]]   # Second sequence
    ], dtype=torch.float32)
    actions = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    # Test both models
    policy_output = policy(timesteps, states, actions)
    critic_output = critic(timesteps, states, actions)

    # Check outputs
    assert policy_output.shape == (batch_size, act_dim), "Policy output shape incorrect"
    assert critic_output.shape == (batch_size, 1), "Critic output shape incorrect"

    print("Simple environment tests passed!")

if __name__ == "__main__":
    print("Running transformer model tests...")
    test_transformer_policy()
    test_transformer_critic()
    test_with_simple_env()
    print("All tests passed!") 