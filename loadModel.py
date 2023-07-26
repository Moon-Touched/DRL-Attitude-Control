import os
import winsound
from stable_baselines3 import PPO
import faulty_env
import numpy as np
import matplotlib.pyplot as plt

env = faulty_env.SpaceCraftEnv1()
model = PPO.load("model/2400000.zip")
obs, _ = env.reset()
obsHistory = np.array([])
torqueHistory = np.array([])
N = 6000
for i in range(N):
    if len(obs) < 6:
        obs = obs[0]
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    if i == 0:
        obsHistory = obs
        torqueHistory = 0.01 * action
    else:
        obsHistory = np.vstack([obsHistory, obs])
        torqueHistory = np.vstack([torqueHistory, 0.01 * action])

timeAxis = np.linspace(0, N / 100, N)
plt.figure()
for idx in range(3):
    plt.plot(timeAxis, obsHistory[:, idx], label="sigma" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("MRP Error")

plt.figure()
for idx in range(3, 6):
    plt.plot(timeAxis, obsHistory[:, idx], label="omega" + str(idx - 2))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity")

plt.figure()
for idx in range(4):
    plt.plot(timeAxis, torqueHistory[:, idx], label="T" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Torques")
plt.show()

duration = 1000
frequency = 500
winsound.Beep(frequency, duration)
