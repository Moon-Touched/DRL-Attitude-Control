import os
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from stable_baselines3 import PPO
from env import MyEnv
from dynamic import DynamicModel


def load(path: str):
    env = MyEnv()
    model = PPO.load(path)
    obs, _ = env.reset()
    for i in range(6000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    quat_history, omega_history, error_history, torque_history = (
        env.model.quat_history,
        env.model.omega_history,
        env.model.error_history,
        env.model.torque_history,
    )
    env.model.draw(error_format="angle")


load("C:/training/LR0.0003/model/2420736.zip")
