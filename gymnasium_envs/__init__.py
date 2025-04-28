
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
    id="SimpleNotiDangerZoneLunarLander",
    entry_point="gymnasium_envs.envs.lunar_lander:SimpleNotiDangerZoneLunarLander",
)

register(
    id="ComplexNotiDangerZoneLunarLander",
    entry_point="gymnasium_envs.envs.lunar_lander:ComplexNotiDangerZoneLunarLander",
)
