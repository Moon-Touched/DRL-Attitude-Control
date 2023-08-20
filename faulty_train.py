import os
import winsound

import pandas as pd
from stable_baselines3 import PPO

import faulty_env
from stable_baselines3.common.callbacks import EvalCallback


def train(path: str, num_timestep: int = 6000, num_episode: int = 200, env_code: int = 1, learning_rate=0.0003):
    models_dir = f"{path}/Env{env_code}_LR{learning_rate}/model"
    logdir = f"{path}/Env{env_code}_LR{learning_rate}/logs"
    name = f"env{env_code}_LR{learning_rate}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if env_code == 1:
        env = faulty_env.SpaceCraftEnv1()
    elif env_code == 2:
        env = faulty_env.SpaceCraftEnv2()
    elif env_code == 3:
        env = faulty_env.SpaceCraftEnv3()
    env.reset()

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
    for i in range(num_episode):
        model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=name)
        model.save(f"{models_dir}/{model.num_timesteps}")
    return


train(path="D:/training/Env3_LR0.0003", env_code=3)
