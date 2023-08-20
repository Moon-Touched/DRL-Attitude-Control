import os
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, unitTestSupport, RigidBodyKinematics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import winsound
from stable_baselines3 import PPO
import faulty_env


def DesinedCaseTest(model_name: str = "", env_code: int = 1):
    a = [-np.pi / 4, 0, np.pi / 4]
    wheelNum = [-1, 0, 1, 2, 3]
    time = [1000, 2000, 3000]
    targetAtt = []
    for x in a:
        for y in a:
            for z in a:
                if not (x == 0 and y == 0 and z == 0):
                    targetAtt.append(np.array([x, y, z]))

    model = PPO.load(f"saved_model/{model_name}")
    for t in time:
        for n in wheelNum:
            error = []
            for r in targetAtt:
                if env_code == 1:
                    env = faulty_env.TestEnv1()
                elif env_code == 2:
                    env = faulty_env.TestEnv2(attRef=RigidBodyKinematics.euler1232MRP(r), wheelNum=n, faultTime=t)
                elif env_code == 3:
                    env = faulty_env.TestEnv3(attRef=RigidBodyKinematics.euler1232MRP(r), wheelNum=n, faultTime=t)
                obs, _ = env.reset()

                N = 6000
                for i in range(N):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, truncated, info = env.step(action)
                    if i == 5999:
                        error.append((r, 4 * np.arctan(np.linalg.norm(obs[0:3]))))
            df = pd.DataFrame(error)
            df.to_csv(f"time={t}_wheelNum={n}.csv", index=False)


def RandomTest(num=100, model_name: str = "", env_code: int = 1):
    model = PPO.load(f"saved_model/{model_name}")
    error = []
    for i in range(num):
        print(i)
        r = np.random.uniform(low=-1, high=1, size=3)
        t = np.random.randint(low=0, high=3000)
        n = np.random.randint(low=0, high=4)

        if env_code == 1:
            env = faulty_env.TestEnv1()
        elif env_code == 2:
            env = faulty_env.TestEnv2(attRef=RigidBodyKinematics.euler1232MRP(r), wheelNum=n, faultTime=t)
        elif env_code == 3:
            env = faulty_env.TestEnv3(attRef=RigidBodyKinematics.euler1232MRP(r), wheelNum=n, faultTime=t)
        obs, _ = env.reset()

        for step in range(6000):
            action, _states = model.predict(obs)
            obs, rewards, dones, truncated, info = env.step(action)
            if step == 5999:
                error.append(4 * np.arctan(np.linalg.norm(obs[0:3])))
    df = pd.DataFrame(error)
    df.to_csv(f"time={t}_wheelNum={n}.csv", index=False)


def LoadModel(attref=[], wheelNum=-1, faultTime=0, model_name: str = "", env_code: int = 1):
    model = PPO.load(f"saved_model/{model_name}")
    error = 0
    N = 6000
    obsHistory = np.array([])
    if env_code == 1:
        env = faulty_env.TestEnv1()
    elif env_code == 2:
        env = faulty_env.TestEnv2(attRef=RigidBodyKinematics.euler1232MRP(attref), wheelNum=wheelNum, faultTime=faultTime)
    elif env_code == 3:
        env = faulty_env.TestEnv3(attRef=RigidBodyKinematics.euler1232MRP(attref), wheelNum=wheelNum, faultTime=faultTime)
    obs, _ = env.reset()

    for step in range(6000):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        if step == 0:
            obsHistory = obs
        else:
            obsHistory = np.vstack([obsHistory, obs])
        if step == 5999:
            error = 4 * np.arctan(np.linalg.norm(obs[0:3]))

    error = np.rad2deg(error)
    timeAxis = np.linspace(0, N / 100, N)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    for idx in range(3):
        axes[0].plot(timeAxis, obsHistory[:, idx], label="$\sigma_" + str(idx + 1) + "$")
        axes[0].legend(loc="lower right")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("MRP Error")
        axes[0].set_title(f"Reward function 2, attitude error={error:.2f} degrees")

    for idx in range(3, 6):
        axes[1].plot(timeAxis, obsHistory[:, idx], label="$\omega_" + str(idx - 2) + "$")
        axes[1].legend(loc="lower right")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Angular Velocity")
    plt.show()


# DesinedCaseTest(model_name="R3.zip", env_code=3)
# RandomTest(num=100, model_name="R3.zip")
LoadModel([-np.pi / 4, np.pi / 4, np.pi / 4], 1, 3000, "R2.zip", 2)
