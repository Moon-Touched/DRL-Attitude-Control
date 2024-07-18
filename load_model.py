import os
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from stable_baselines3 import PPO
from train import TrainEnv1


def load(path: str):
    env = TrainEnv1()
    model = PPO.load(path)
    obs, _ = env.reset()
    for i in range(6000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    MRP_history = env.model.MRP_history
    erroe_MRP_history = env.model.erroe_MRP_history
    error_angle_history = env.model.error_angle_history
    omega_history = env.model.omega_history
    omega_dot_history = env.model.omega_dot_history
    plt.plot(MRP_history)
    plt.show()
    plt.plot(erroe_MRP_history)
    plt.show()
    plt.plot(error_angle_history)
    plt.show()
    plt.plot(omega_history)
    plt.show()
    plt.plot(omega_dot_history)
    plt.show()


load("C:\\training\\Env2_LR0.0003\\model\\2260992.zip")
