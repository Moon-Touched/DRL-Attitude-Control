from stable_baselines3.common.env_checker import check_env
from MyEnv import SpaceCraftEnv

env = SpaceCraftEnv(reference=[0.1, 0.1, 0.1])
check_env(env)
