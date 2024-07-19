import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import Tools


def load(path: str):
    model = PPO.load(path)
    env = Tools.make_env("benv01", 1, False, "wheel")
    obs, _ = env.reset()
    for i in range(6000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
    fig, axes = Tools.plot_history(env.model)
    fig.suptitle(f"fault time: {env.fault_time}, wheel num: {env.wheel_num}")
    plt.show()


def evaluate(path: str, episode_num=10):
    model = PPO.load(path)
    reward_history = []
    ss_error_history = []
    for k in range(episode_num):
        episode_reward = 0
        env = Tools.make_env("menv01", 1, False, "wheel")
        obs, _ = env.reset()
        for i in range(6000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        ss_error = env.model.error_angle_history[-1]
        reward_history.append(episode_reward)
        ss_error_history.append(ss_error)
        print(f"episode: {k}/{episode_num}, reward: {episode_reward}, error: {ss_error}")
    plt.plot(reward_history)
    plt.show()
    plt.plot(ss_error_history)
    plt.show()


evaluate("C:\\training\\train_Basilisk\\menv01_False_wheel\\model\\2916352.zip")
