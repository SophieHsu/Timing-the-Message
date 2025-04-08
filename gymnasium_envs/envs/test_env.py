import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Simple1DEnv(gym.Env):
    """
    A simple 1D environment where the agent needs to move to a target position.
    State: [position, target_position]
    Action: 0 (move left) or 1 (move right)
    Reward: -1 for each step, +10 for reaching target
    """
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: left, 1: right
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0]),  # [position, target]
            high=np.array([10.0, 10.0]),
            dtype=np.float32
        )
        
        # Environment parameters
        self.max_steps = 100
        self.step_size = 0.5
        self.render_mode = render_mode
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly initialize position and target
        self.position = self.np_random.uniform(-10.0, 10.0)
        self.target = self.np_random.uniform(-10.0, 10.0)
        self.steps = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.steps += 1
        
        # Move based on action
        if action == 0:  # left
            self.position -= self.step_size
        else:  # right
            self.position += self.step_size
            
        # Clip position to bounds
        self.position = np.clip(self.position, -10.0, 10.0)
        
        # Calculate reward
        distance = abs(self.position - self.target)
        reward = -1.0  # small penalty for each step
        if distance < self.step_size:
            reward = (1.0 - 1.0 / (self.max_steps - self.steps)) * 10.0  # reward for reaching target
            
        # Check if episode should end
        terminated = distance < self.step_size
        truncated = self.steps >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self):
        return np.array([self.position, self.target], dtype=np.float32)
    
    def render(self):
        if self.render_mode is None:
            return
        
        # Simple ASCII rendering
        width = 40
        line = [' '] * width
        pos_idx = min(width-1, int((self.position + 10) * width / 20))
        target_idx = min(width-1, int((self.target + 10) * width / 20))
        
        line[pos_idx] = 'A'  # Agent
        line[target_idx] = 'T'  # Target
        
        print(''.join(line)) 