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

    def __init__(self, render_mode="console", reference: list = [0, 0, 0]):
        super(SpaceCraftEnv1, self).__init__()

        self.render_mode = render_mode
        self.reference = reference
        self.attError = [0, 0, 0]
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.action = [0, 0, 0, 0]
        self.observation = []
        self.stepCount = 0
        # Torques in x, y, z.
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Attitude errors and angular velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.attError = RigidBodyKinematics.subMRP(np.array([0, 0, 0]), -np.array(self.reference)).tolist()
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.observation = np.array(self.attError + self.omega).astype(np.float32)
        self.stepCount = 0
        return self.observation, {}  # empty info dict

    def step(self, action):
        done = False
        truncated = False
        self.stepCount = self.stepCount + 1
        self.action = 0.01 * action
        self.sigma, self.omega = DynSpacecraft(0.01, 0.01, self.sigma, self.omega, self.action)
        self.attError = RigidBodyKinematics.subMRP(np.array(self.sigma), -np.array(self.reference)).tolist()
        self.observation = np.array(self.attError + self.omega).astype(np.float32)

        attSquare = 0
        omegaSquare = 0
        for i in range(3):
            attSquare = attSquare + np.square(self.observation[i])
        for i in range(3, 6):
            omegaSquare = omegaSquare + np.square(self.observation[i])
        r1 = -(attSquare + omegaSquare)

        r2 = 0
        if attSquare < 0.005:
            r2 = 10
            print("Right attitude")

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

#
#去掉1中超过最大角速度停止训练的条件
#
class SpaceCraftEnv1_5(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console", reference: list = [0, 0, 0]):
        super(SpaceCraftEnv1_5, self).__init__()

        self.render_mode = render_mode
        self.reference = reference
        self.attError = [0, 0, 0]
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.action = [0, 0, 0, 0]
        self.observation = []
        self.stepCount = 0
        # Torques in x, y, z.
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Attitude errors and angular velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.attError = RigidBodyKinematics.subMRP(np.array([0, 0, 0]), -np.array(self.reference)).tolist()
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.observation = np.array(self.attError + self.omega).astype(np.float32)
        self.stepCount = 0
        return self.observation, {}  # empty info dict

    def step(self, action):
        done = False
        truncated = False
        self.stepCount = self.stepCount + 1
        self.action = 0.01 * action
        self.sigma, self.omega = DynSpacecraft(0.01, 0.01, self.sigma, self.omega, self.action)
        self.attError = RigidBodyKinematics.subMRP(np.array(self.sigma), -np.array(self.reference)).tolist()
        self.observation = np.array(self.attError + self.omega).astype(np.float32)

        attSquare = 0
        omegaSquare = 0
        for i in range(3):
            attSquare = attSquare + np.square(self.observation[i])
        for i in range(3, 6):
            omegaSquare = omegaSquare + np.square(self.observation[i])
        r1 = -(attSquare + omegaSquare)

        r2 = 0
        if attSquare < 0.005:
            r2 = 10
            print("Right attitude")

        r3 = 0
        if np.abs(self.omega[0]) > 1 or np.abs(self.omega[1]) > 1 or np.abs(self.omega[2]) > 1:
            r3 = -100
            done = True

        self.reward = r1 + r2 + r3

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        return self.observation, self.reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass



class SpaceCraftEnv2(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console", reference: list = [0, 0, 0]):
        super(SpaceCraftEnv2, self).__init__()

        self.render_mode = render_mode
        self.reference = reference
        self.attError = [0, 0, 0]
        self.preAttError = [0, 0, 0]
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.action = [0, 0, 0, 0]
        self.observation = []
        self.stepCount = 0
        # Torques in x, y, z.
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # Attitude errors and angular velocity
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.attError = RigidBodyKinematics.subMRP(np.array([0, 0, 0]), -np.array(self.reference)).tolist()
        self.preAttError = [0, 0, 0]
        self.sigma = [0, 0, 0]
        self.omega = [0, 0, 0]
        self.reward = 0
        self.observation = np.array(self.attError + self.omega).astype(np.float32)
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
        self.observation = np.array(self.attError + self.omega).astype(np.float32)

        pre = RigidBodyKinematics.MRP2Euler123(self.preAttError)
        cur = RigidBodyKinematics.MRP2Euler123(self.attError)
        rate = pre - cur
        r1 = 0
        for i in range(3):
            r1 = r1 + rate[0]

        attSquare = 0
        for i in range(3):
            attSquare = attSquare + np.square(self.observation[i])
        r2 = 0
        if attSquare < 0.005:
            r2 = 10
            print("Right attitude")

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


class LoggerCallback(BaseCallback):
    def __init__(self, verbose=0, roundCount=0):
        super().__init__(verbose)
        self.totalReward = 0
        self.roundCount = roundCount

    def _on_step(self) -> bool:
        self.totalReward = self.totalReward + self.training_env.envs[0].reward
        for idx in range(3):
            self.logger.record(f"Round{self.roundCount}_attError{idx+1}", self.training_env.envs[0].attError[idx])
        for idx in range(3):
            self.logger.record(f"Round{self.roundCount}_omega{idx+1}", self.training_env.envs[0].omega[idx])
        for idx in range(4):
            self.logger.record(f"Round{self.roundCount}_T{idx+1}", self.training_env.envs[0].action[idx])
        self.logger.dump(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.logger.record("epispde length", self.training_env.envs[0].stepCount)
        self.logger.record("epispde reward", self.totalReward)
        self.logger.dump(self.num_timesteps)
        pass


class StateSaveCallback(BaseCallback):
    def __init__(self, verbose=0, roundCount=0):
        super(StateSaveCallback, self).__init__(verbose)
        self.cnt = roundCount
        self.states = []

    def _on_step(self) -> bool:
        self.states.append(self.training_env.envs[0].sigma + self.training_env.envs[0].omega + self.training_env.envs[0].action.tolist())
        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.states, columns=["Attitude_1", "Attitude_2", "Attitude_3", "omega_1", "omega_2", "omega_3", "Torque_1", "Torque_2", "Torque_3", "Torque_4"])
        filename = f"./stateHistory/round_{self.cnt+1}.csv"
        df.to_csv(filename, index=False)
        pass
