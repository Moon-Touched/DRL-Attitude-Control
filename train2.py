import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from stable_baselines3 import PPO
from env import MyEnv


def train2(path: str, num_timestep: int = 6000, num_episode: int = 2000, learning_rate=0.0003):
    models_dir = f"{path}/LR{learning_rate}/model"
    logdir = f"{path}/LR{learning_rate}/logs"
    name = f"LR{learning_rate}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = MyEnv()
    env.reset()

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
    for i in range(num_episode):
        print(f"num:{i}")
        model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=name)
        model.save(f"{models_dir}/{model.num_timesteps}")
    return


train2(path="C:/training/env2", num_timestep=6000, num_episode=2000, learning_rate=0.0003)
