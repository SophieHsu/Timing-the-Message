
from gymnasium.envs.registration import register

register(
    id="NotiLunarLander",
    entry_point="gymnasium_envs.envs.lunar_lander:NotiLunarLander",
)

register(
    id="LargeRewardNotiLunarLander",
    entry_point="gymnasium_envs.envs.lunar_lander:LargeRewardNotiLunarLander",
)

register(
    id="DangerZoneLunarLander",
    entry_point="gymnasium_envs.envs.lunar_lander:DangerZoneLunarLander",
)

register(
    id="NotiHighway",
    entry_point="gymnasium_envs.envs.highway:NotiHighway",
)

register(
    id="NotiHighwayFast",
    entry_point="gymnasium_envs.envs.highway:NotiHighwayFast",
)

register(
    id="Simple1DEnv",
    entry_point="gymnasium_envs.envs.test_env:Simple1DEnv",
)
