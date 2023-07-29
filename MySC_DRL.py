import os

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO

# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import attTrackingError, inertial3D, rwMotorTorque
from Basilisk.simulation import reactionWheelStateEffector, simpleNav, spacecraft, extForceTorque
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, unitTestSupport, RigidBodyKinematics

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


def DynSpacecraft(attref, totalTime, timestep):
    # region Simulation Parameters
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    simulationTime = macros.sec2nano(totalTime)
    simulationTimeStep = macros.sec2nano(timestep)
    # endregion

    # region create the simulation process and task
    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))
    # endregion

    # region spacecraft  properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "mySpacecraft"
    I = [0.025, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.065]
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    scSim.AddModelToTask(simTaskName, scObject, None, 10)
    # endregion

    # Reaction wheels
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels
    RW1 = rwFactory.create("Honeywell_HR16", [1, 0, 0], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
    RW2 = rwFactory.create("Honeywell_HR16", [0, 1, 0], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
    RW3 = rwFactory.create("Honeywell_HR16", [0, 0, 1], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
    RW4 = rwFactory.create("Honeywell_HR16", [1, 1, 1], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)

    numRW = rwFactory.getNumOfDevices()

    #  RW  container
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
    scSim.AddModelToTask(simTaskName, rwStateEffector, None, 20)

    # setup the controller
    controller = Controller(attref)
    controller.ModelTag = "myController"
    scSim.AddModelToTask(simTaskName, controller, None, 1)

    # link messages
    controller.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(controller.wheelTorqueOutMsg)

    stateLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, stateLog)

    attErrorLog = controller.attErrorOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, attErrorLog)

    rwMotorLog = controller.wheelTorqueOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, rwMotorLog)

    # cmdTorqueLog = controller.axisTorqueOutMsg.recorder()
    # scSim.AddModelToTask(simTaskName, cmdTorqueLog)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ShowExecutionOrder()
    scSim.ExecuteSimulation()

    timeAxis = attErrorLog.times()
    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, attErrorLog.sigma_BR[:, idx], color=unitTestSupport.getLineColor(idx, 3), label="sigma" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("MRP Error")

    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.omega_BN_B[:, idx], color=unitTestSupport.getLineColor(idx, 3), label="omega" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Current angular velocity [r/s]")

    plt.figure()
    for idx in range(numRW):
        plt.plot(timeAxis * macros.NANO2SEC, rwMotorLog.motorTorque[:, idx], label="RW" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("RW Torque")

    # hingiadaiojlaaijianif
    # plt.figure()
    # for idx in range(3):
    #     plt.plot(timeAxis * macros.NANO2SEC, cmdTorqueLog.torqueRequestBody[:, idx], label="T" + str(idx + 1))
    # plt.legend(loc="lower right")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Axis Torque")

    plt.show()

    return


class Controller(sysModel.SysModel):
    def __init__(self, attRef):
        super(Controller, self).__init__()

        self.attRef = np.array(attRef)
        self.stateInMsg = messaging.SCStatesMsgReader()
        self.wheelTorqueOutMsg = messaging.ArrayMotorTorqueMsg()
        self.attErrorOutMsg = messaging.AttGuidMsg()
        self.model = PPO.load("saved_model/R2.zip")

    def Reset(self, CurrentSimNanos):
        return

    def UpdateState(self, CurrentSimNanos):
        stateInMsgBuffer = self.stateInMsg()
        curSigma_BN = np.array(stateInMsgBuffer.sigma_BN)
        curOmega_BN_B = np.array(stateInMsgBuffer.omega_BN_B)
        wheelTorqueOutMsgBuffer = messaging.ArrayMotorTorqueMsgPayload()
        attErrorOutMsgBuffer = messaging.AttGuidMsgPayload()
        print("2")
        attError = RigidBodyKinematics.subMRP(curSigma_BN, -self.attRef)
        print(np.array(attError.tolist() + curOmega_BN_B.tolist()).astype(np.float32))
        obs = np.array(attError.tolist() + curOmega_BN_B.tolist()).astype(np.float16)
        action, _ = self.model.predict(obs)
        print(action)

        attErrorOutMsgBuffer.sigma_BR = attError.tolist()

        wheelTorqueOutMsgBuffer.motorTorque = action.tolist()

        self.wheelTorqueOutMsg.write(wheelTorqueOutMsgBuffer, CurrentSimNanos, self.moduleID)
        self.attErrorOutMsg.write(attErrorOutMsgBuffer, CurrentSimNanos, self.moduleID)

        return


if __name__ == "__main__":
    # reference, simulation time, timestep
    DynSpacecraft(np.array([0.5, 0.5, 0.5]), 60, 0.01)
