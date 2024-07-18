import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from stable_baselines3 import PPO
from Envs import BasiliskEnv
import matplotlib.pyplot as plt
import Tools


def load(path: str):
    model = PPO.load(path)
    env = BasiliskEnv(faulty=True, torque_mode="wheel")
    obs, _ = env.reset()
    for i in range(6000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    fig, axes = Tools.plot_history(env.model)
    fig.suptitle(f"fault time: {env.fault_time}, wheel num: {env.wheel_num}")
    plt.show()


load("E:\\training\\train_Basilisk\\env01_False_wheel\\model\\3473408.zip")
