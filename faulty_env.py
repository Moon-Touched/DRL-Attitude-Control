import os

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Basilisk import __path__
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import attTrackingError, inertial3D, rwMotorTorque
from Basilisk.simulation import reactionWheelStateEffector, simpleNav, spacecraft, extForceTorque
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, unitTestSupport, RigidBodyKinematics


bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


def DynSpacecraft(totalTime, timestep, currentMRP: list, currentOmega: list, wheelTorque: list):
    # Simulation Parameters
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    simulationTime = macros.sec2nano(totalTime)
    simulationTimeStep = macros.sec2nano(timestep)

    # create the simulation process and task
    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    #  spacecraft  properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "mySpacecraft"
    I = [0.025, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.065]
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    scObject.hub.sigma_BNInit = currentMRP
    scObject.hub.omega_BN_BInit = currentOmega
    scSim.AddModelToTask(simTaskName, scObject, None, 1)

    #
    # Reaction wheels
    #
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels
    RW1 = rwFactory.create("Honeywell_HR16", [1, 0, 0], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
    RW2 = rwFactory.create("Honeywell_HR16", [0, 1, 0], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
    RW3 = rwFactory.create("Honeywell_HR16", [0, 0, 1], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
    RW4 = rwFactory.create("Honeywell_HR16", [1, 1, 1], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)

    numRW = rwFactory.getNumOfDevices()

    #
    # RW  container
    #
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
    scSim.AddModelToTask(simTaskName, rwStateEffector, None, 2)

    #
    # Stand alone msg
    #
    rwParamMsg = rwFactory.getConfigMessage()
    wheelTorqueMsg = messaging.ArrayMotorTorqueMsg()
    wheelTorqueBuffer = messaging.ArrayMotorTorqueMsgPayload()
    wheelTorqueBuffer.motorTorque = wheelTorque
    wheelTorqueMsg.write(wheelTorqueBuffer)

    #
    # link messages
    #
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(wheelTorqueMsg)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    # scSim.ShowExecutionOrder()
    scSim.ExecuteSimulation()

    scState = scObject.scStateOutMsg.read()
    scSigma = scState.sigma_BN
    scOmega = scState.omega_BN_B
    # print(scSigma, scOmega)
    return scSigma, scOmega


class SpaceCraftEnv1(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console"):
        super(SpaceCraftEnv1, self).__init__()

        self.render_mode = render_mode
        self.reference = [0, 0, 0]
        self.attError = [0, 0, 0]
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.action = [0, 0, 0, 0]
        self.observation = []
        self.stepCount = 0
        self.faultTime = 0
        self.wheelNum = 0
        # Torques in x, y, z.
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float16)

        # Attitude errors and angular velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float16)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reference = np.random.uniform(low=-1, high=1, size=3)
        self.faultTime = np.random.randint(low=0, high=3000)
        self.wheelNum = np.random.randint(low=0, high=4)
        self.attError = RigidBodyKinematics.subMRP(np.array([0, 0, 0]), -np.array(self.reference)).tolist()
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.observation = np.array(self.attError + self.omega).astype(np.float16)
        self.stepCount = 0
        print(f"reference:{self.reference}")
        print(f"faultTime:{self.faultTime}")
        print(f"wheelNum:{self.wheelNum}")
        return self.observation, {}

    def step(self, action):
        done = False
        truncated = False
        self.stepCount = self.stepCount + 1
        self.action = 0.01 * action
        if self.stepCount > self.faultTime:
            self.action[self.wheelNum] = 0
        self.sigma, self.omega = DynSpacecraft(0.01, 0.01, self.sigma, self.omega, self.action)
        self.attError = RigidBodyKinematics.subMRP(np.array(self.sigma), -np.array(self.reference)).tolist()
        self.observation = np.array(self.attError + self.omega).astype(np.float16)

        curError = 4 * np.arctan(np.linalg.norm(self.attError))
        for i in range(3, 6):
            omegaSum = omegaSum + np.abs(self.observation[i])
        r1 = -(curError + omegaSum)

        r2 = 0
        if curError < 0.0043633:
            r2 = 10

        r3 = 0
        if np.abs(self.omega[0]) > 1 or np.abs(self.omega[1]) > 1 or np.abs(self.omega[2]) > 1:
            r3 = -100
            done = True

        self.reward = r1 + r2 + r3

        if self.stepCount > 6000:
            truncated = True
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        return self.observation, self.reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass


class SpaceCraftEnv2(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console"):
        super(SpaceCraftEnv2, self).__init__()

        self.render_mode = render_mode
        self.reference = [0, 0, 0]
        self.attError = [0, 0, 0]
        self.preAttError = [0, 0, 0]
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.action = [0, 0, 0, 0]
        self.observation = []
        self.stepCount = 0
        # Torques in x, y, z.
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float16)

        # Attitude errors and angular velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float16)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.attError = RigidBodyKinematics.subMRP(np.array([0, 0, 0]), -np.array(self.reference)).tolist()
        self.preAttError = [0, 0, 0]
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.observation = np.array(self.attError + self.omega).astype(np.float16)
        self.stepCount = 0
        return self.observation, {}  # empty info dict

    def step(self, action):
        done = False
        truncated = False
        self.stepCount = self.stepCount + 1
        self.action = 0.01 * action
        self.sigma, self.omega = DynSpacecraft(0.01, 0.01, self.sigma, self.omega, self.action)
        self.preAttError = self.attError
        self.attError = RigidBodyKinematics.subMRP(np.array(self.sigma), -np.array(self.reference)).tolist()
        self.observation = np.array(self.attError + self.omega).astype(np.float16)

        preError = 4 * np.arctan(np.linalg.norm(self.preAttError))
        curError = 4 * np.arctan(np.linalg.norm(self.attError))
        rate = preError - curError
        r1 = 0
        if self.stepCount == 6000 and curError < 0.0043633:
            r1 = 1

        r2 = (preError - curError) / np.pi

        r3 = 0
        if np.abs(self.omega[0]) > 1 or np.abs(self.omega[1]) > 1 or np.abs(self.omega[2]) > 1:
            r3 = -1
            done = True

        self.reward = r1 + r2 + r3

        if self.stepCount > 6000:
            truncated = True
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        return self.observation, self.reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass
