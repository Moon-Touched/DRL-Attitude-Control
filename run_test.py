import os
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, unitTestSupport, RigidBodyKinematics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import winsound
from stable_baselines3 import PPO
import faulty_env


def DesinedCaseTest():
    a = [-np.pi / 4, 0, np.pi / 4]
    wheelNum = [-1, 0, 1, 2, 3]
    time = [1000, 2000, 3000]
    targetAtt = []
    for x in a:
        for y in a:
            for z in a:
                if not (x == 0 and y == 0 and z == 0):
                    targetAtt.append(np.array([x, y, z]))

    model = PPO.load("saved_model/R2.zip")
    for t in time:
        for n in wheelNum:
            error = []
            for r in targetAtt:
                env = faulty_env.TestEnv2(attRef=RigidBodyKinematics.euler1232MRP(r), wheelNum=n, faultTime=t)
                obs, _ = env.reset()
                N = 6000
                for i in range(N):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, truncated, info = env.step(action)
                    if i == 5999:
                        error.append((r, 4 * np.arctan(np.linalg.norm(obs[0:3]))))
            df = pd.DataFrame(error)
            df.to_csv(f"time={t}_wheelNum={n}.csv", index=False)


def RandomTest(num=100):
    model = PPO.load("saved_model/R2.zip")
    error = 0
    for i in range(num):
        print(i)
        r = np.random.uniform(low=-1, high=1, size=3)
        t = np.random.randint(low=0, high=3000)
        n = np.random.randint(low=0, high=4)

        env = faulty_env.TestEnv2(attRef=r, wheelNum=n, faultTime=t)
        obs, _ = env.reset()
        for step in range(6000):
            action, _states = model.predict(obs)
            obs, rewards, dones, truncated, info = env.step(action)
            if step == 5999:
                error = error + 4 * np.arctan(np.linalg.norm(obs[0:3]))
    aveError = error / num
    print(aveError)


RandomTest(num=500)
