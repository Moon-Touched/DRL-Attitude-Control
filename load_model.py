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
    fig = plt.figure()

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(quat_history, label=["w", "x", "y", "z"])
    ax.set_title("Quaternion History")
    ax.legend()

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(error_history, label=["error_x", "error_y", "error_z"])
    ax.set_title("Error History")
    ax.legend()

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(omega_history, label=["omega_x", "omega_y", "omega_z"])
    ax.set_title("Angular Velocity History")
    ax.legend()

    plt.show()


load("C:/training/LR0.0003/model/61440.zip")
