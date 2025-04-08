
from gymnasium.envs.registration import register

register(
    id="gymnasium_envs/NotiLunarLander",
    entry_point="gymnasium_envs.envs.lunar_lander:NotiLunarLander",
)

register(
    id="gymnasium_envs/LargeRewardNotiLunarLander",
    entry_point="gymnasium_envs.envs.lunar_lander:LargeRewardNotiLunarLander",
)

register(
    id="gymnasium_envs/NotiHighway",
    entry_point="gymnasium_envs.envs.highway:NotiHighway",
)

register(
    id="gymnasium_envs/NotiHighwayFast",
    entry_point="gymnasium_envs.envs.highway:NotiHighwayFast",
)

register(
    id="gymnasium_envs/Simple1DEnv",
    entry_point="gymnasium_envs.envs.test_env:Simple1DEnv",
)
