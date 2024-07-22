import os, multiprocessing, Tools

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv


def train(
    path: str,
    env_name: str,
    num_timestep: int,
    num_episode: int = 2000,
    env_num: int = 4,
    vec_env_cls=DummyVecEnv,
    faulty: bool = False,
    torque_mode: str = "wheel",
    device: str = "cuda",
):

    train_name = f"{env_name}_{faulty}_{torque_mode}"
    models_dir = f"{path}/{train_name}/model"
    logdir = f"{path}/{train_name}/logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = Tools.make_env(env_name=env_name, env_num=env_num, faulty=faulty, torque_mode=torque_mode, vec_env_cls=vec_env_cls)

    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log=logdir, device=device)
    for i in range(num_episode):
        print("**********************************")
        print(f"num:{i}")
        model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=train_name)
        model.save(f"{models_dir}/{model.num_timesteps}")
    return


if __name__ == "__main__":
    train(
        path="E:\\training\\train_Basilisk",
        env_name="benv05",
        num_timestep=50000,
        num_episode=1000,
        env_num=16,
        faulty=True,
        torque_mode="wheel",
        device="cuda",
    )
