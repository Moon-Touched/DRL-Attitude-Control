import os
import winsound

import pandas as pd
from stable_baselines3 import PPO

import faulty_env
from stable_baselines3.common.callbacks import EvalCallback


def continueTrain(path: str, filename: str = "", num_timestep: int = 6000, num_episode: int = 200, env_code: int = 1, learning_rate=0.0003):
    models_dir = f"{path}/model"
    logdir = f"{path}/logs"
    name = f"env{env_code}_LR{learning_rate}"

    if env_code == 1:
        env = faulty_env.SpaceCraftEnv1()
    elif env_code == 2:
        env = faulty_env.SpaceCraftEnv2()
    elif env_code == 3:
        env = faulty_env.SpaceCraftEnv3()
    env.reset()

    model = PPO.load(path=f"{models_dir}/{filename}", env=env, tensorboard_log=logdir)
    for i in range(num_episode):
        model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=name)
        model.save(f"{models_dir}/{model.num_timesteps}")


continueTrain("D:/training/Env3_LR0.0003", "2457600.zip", env_code=3)
