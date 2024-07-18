import os, multiprocessing

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from Envs import BasiliskEnv, BasiliskRWEnv


class TrainEnv1(BasiliskEnv):
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


class TrainEnv2(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (10 - cur_error) * (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if cur_error < 0.0043633 and np.linalg.norm(self.model.cur_omega) < 0.001:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class TrainEnv3(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 1 / (cur_error + 1)

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class TrainEnv4(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / pre_error

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if cur_error < 0.0043633:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


def train(
    path: str,
    env_name: str,
    num_timestep: int,
    num_episode: int = 2000,
    env_num=4,
    vec_env_cls=DummyVecEnv,
    faulty=False,
    torque_mode: str = "wheel",
    device="cuda",
):
    """Train the model.

    Args:
        path (str): The path to save the trained models and logs.
        env_name (str): The name of the environment to train on.
        num_timestep (int): The total number of timesteps to train for.
        num_episode (int, optional): The number of episodes to train for. Defaults to 2000.
        env_num (int, optional): The number of parallel environments. Defaults to 4.
        vec_env_cls (class, optional): The class for creating vectorized environments. Defaults to DummyVecEnv.
        faulty (bool, optional): Whether to use a faulty environment. Defaults to False.
        torque_mode (str, optional): The torque mode to use. Defaults to "wheel".
        device (str, optional): The device to use for training. Defaults to "cuda".

    Returns:
        None
    """

    train_name = f"{env_name}_{faulty}_{torque_mode}"
    models_dir = f"{path}/{train_name}/model"
    logdir = f"{path}/{train_name}/logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if env_num == 1:
        if env_name == "env01":
            env = TrainEnv1(faulty=faulty, torque_mode=torque_mode)
        elif env_name == "env02":
            env = TrainEnv2(faulty=faulty, torque_mode=torque_mode)
        elif env_name == "env03":
            env = TrainEnv3(faulty=faulty, torque_mode=torque_mode)
        elif env_name == "env04":
            env = TrainEnv4(faulty=faulty, torque_mode=torque_mode)
        env.reset()
    else:
        if env_name == "env01":
            env = make_vec_env(TrainEnv1, n_envs=env_num, env_kwargs={"faulty": faulty}, vec_env_cls=vec_env_cls)
        elif env_name == "env02":
            env = make_vec_env(TrainEnv2, n_envs=env_num, env_kwargs={"faulty": faulty}, vec_env_cls=vec_env_cls)
        elif env_name == "env03":
            env = make_vec_env(TrainEnv3, n_envs=env_num, env_kwargs={"faulty": faulty}, vec_env_cls=vec_env_cls)
        elif env_name == "env04":
            env = make_vec_env(TrainEnv4, n_envs=env_num, env_kwargs={"faulty": faulty}, vec_env_cls=vec_env_cls)

    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log=logdir, device=device)
    for i in range(num_episode):
        print("**********************************")
        print(f"num:{i}")
        model.learn(total_timesteps=num_timestep, reset_num_timesteps=False, tb_log_name=train_name)
        model.save(f"{models_dir}/{model.num_timesteps}")
    return


# if __name__ == "__main__":
#     train(
#         path="E:\\training\\train_Basilisk",
#         env_name="env04",
#         num_timestep=100000,
#         num_episode=1000,
#         env_num=16,
#         faulty=False,
#         torque_mode="wheel",
#         device="cuda",
#     )
