import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from Envs import BasiliskEnv


class TrainEnv(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if self.step_count == 6000 and cur_error < 0.0043633:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


def train_vec_env(path: str, name: str, num_timestep: int, num_episode: int = 2000, env_num=4, faulty=False):
    models_dir = f"{path}/{name}/model"
    logdir = f"{path}/{name}/logs"
    name = f"{name}_{faulty}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    vec_env = make_vec_env(TrainEnv, n_envs=env_num, env_kwargs={"faulty": faulty})

    model = PPO("MlpPolicy", vec_env, verbose=2, tensorboard_log=logdir, device="cuda")
    for i in range(num_episode):
        print("**********************************")
        print(f"num:{i}")
        model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=name)
        model.save(f"{models_dir}/{model.num_timesteps}")
    return


if __name__ == "__main__":
    train_vec_env(path="E:/train_Basilisk", name="env01", num_timestep=100000, num_episode=2000, env_num=32, faulty=False)
