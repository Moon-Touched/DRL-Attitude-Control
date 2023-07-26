import os
import winsound

import pandas as pd
from stable_baselines3 import PPO

import faulty_env
from stable_baselines3.common.callbacks import EvalCallback

current_directory = os.getcwd()
name = os.path.basename(current_directory)

models_dir = "model"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

num_timestep = 6000
epis = 200

env2 = faulty_env.SpaceCraftEnv2()
env2.reset()

model = PPO("MlpPolicy", env2, verbose=0, tensorboard_log=logdir)
for i in range(epis):
    model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=name)
    model.save(f"{models_dir}/{model.num_timesteps}")

duration = 1000
frequency = 500
winsound.Beep(frequency, duration)
