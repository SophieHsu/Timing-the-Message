
from gymnasium.envs.registration import register

register(
    id="gymnasium_envs/NotiLunarLander",
    entry_point="gymnasium_envs.envs.lunar_lander:NotiLunarLander",
)

register(
    id="gymnasium_envs/NotiHighway",
    entry_point="gymnasium_envs.envs.highway:NotiHighway",
)

register(
    id="gymnasium_envs/NotiHighwayFast",
    entry_point="gymnasium_envs.envs.highway:NotiHighwayFast",
)
