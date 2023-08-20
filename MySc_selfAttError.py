import os

import matplotlib.pyplot as plt
import numpy as np

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
    RW4 = rwFactory.create("Honeywell_HR16", [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)

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

    # Sim Manager
    simManager = SimManager()
    simManager.ModelTag = "SimManager"
    scSim.AddModelToTask(simTaskName, simManager)

    # external torque
    extTorque = extForceTorque.ExtForceTorque()
    extTorque.ModelTag = "externalTorque"
    scObject.addDynamicEffector(extTorque)
    scSim.AddModelToTask(simTaskName, extTorque)

    # rwMotorTorqueConfig = rwMotorTorque.rwMotorTorqueConfig()
    # rwMotorTorqueWrap = scSim.setModelDataWrap(rwMotorTorqueConfig)
    # rwMotorTorqueWrap.ModelTag = "rwMotorTorque"
    # scSim.AddModelToTask(simTaskName, rwMotorTorqueWrap, rwMotorTorqueConfig)

    # controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    # rwMotorTorqueConfig.controlAxes_B = controlAxes_B

    rwParamMsg = rwFactory.getConfigMessage()

    # link messages
    controller.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    # rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(rwParamMsg)
    # rwMotorTorqueConfig.vehControlInMsg.subscribeTo(controller.axisTorqueOutMsg)
    # rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueConfig.rwMotorTorqueOutMsg)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(controller.wheelTorqueOutMsg)
    extTorque.cmdTorqueInMsg.subscribeTo(simManager.extTorqueOutMsg)

    stateLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, stateLog)

    attErrorLog = controller.attErrorOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, attErrorLog)

    # rwMotorLog = rwMotorTorqueConfig.rwMotorTorqueOutMsg.recorder()
    # scSim.AddModelToTask(simTaskName, rwMotorLog)

    cmdTorqueLog = controller.axisTorqueOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, cmdTorqueLog)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ShowExecutionOrder()
    scSim.ExecuteSimulation()

    timeAxis = attErrorLog.times()
    fig, axes = plt.subplots(nrows=1, ncols=3)
    for idx in range(3):
        axes[0].plot(timeAxis * macros.NANO2SEC, attErrorLog.sigma_BR[:, idx], color=unitTestSupport.getLineColor(idx, 3), label=r"$\sigma_" + str(idx + 1) + "$")
    axes[0].legend(loc="lower right")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("MRP Error")

    for idx in range(3):
        axes[1].plot(timeAxis * macros.NANO2SEC, stateLog.omega_BN_B[:, idx], color=unitTestSupport.getLineColor(idx, 3), label=r"$\omega_" + str(idx + 1) + "$")
    axes[1].legend(loc="lower right")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Current angular velocity [r/s]")

    # for idx in range(numRW):
    #     axes[2].plot(timeAxis * macros.NANO2SEC, rwMotorLog.motorTorque[:, idx], label="RW" + str(idx + 1))
    # axes[2].legend(loc="lower right")
    # axes[2].set_xlabel("Time [s]")
    # axes[2].set_ylabel("RW Torque [Nm]")

    # hingiadaiojlaaijianif
    # plt.figure()
    # for idx in range(3):
    #     plt.plot(timeAxis * macros.NANO2SEC, cmdTorqueLog.torqueRequestBody[:, idx], label="T" + str(idx + 1))
    # plt.legend(loc="lower right")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Axis Torque")
    plt.tight_layout()
    plt.show()

    return


class Controller(sysModel.SysModel):
    def __init__(self, attRef):
        super(Controller, self).__init__()
        self.Kp = 0.1
        self.Kd = 0.1
        self.attRef = np.array(attRef)
        self.stateInMsg = messaging.SCStatesMsgReader()
        self.wheelTorqueOutMsg = messaging.ArrayMotorTorqueMsg()
        self.axisTorqueOutMsg = messaging.CmdTorqueBodyMsg()
        self.attErrorOutMsg = messaging.AttGuidMsg()

    def Reset(self, CurrentSimNanos):
        return

    def UpdateState(self, CurrentSimNanos):
        stateInMsgBuffer = self.stateInMsg()
        CurSigma_BN = np.array(stateInMsgBuffer.sigma_BN)
        curOmega_BN_B = np.array(stateInMsgBuffer.omega_BN_B)
        wheelTorqueOutMsgBuffer = messaging.ArrayMotorTorqueMsgPayload()
        axisTorqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()
        attErrorOutMsgBuffer = messaging.AttGuidMsgPayload()

        attError = RigidBodyKinematics.subMRP(CurSigma_BN, -self.attRef)
        attErrorOutMsgBuffer.sigma_BR = attError.tolist()

        axisTorque = (np.array(attError) * self.Kp + np.array(curOmega_BN_B) * self.Kd).tolist()
        # wheelTorque = np.array([0, 0, 0, 0])
        # wheelTorque = (np.array(attError) * self.Kp + np.array(curOmega_BN_B) * self.Kd).tolist()

        axisTorqueOutMsgBuffer.torqueRequestBody = axisTorque
        wheelTorqueOutMsgBuffer.motorTorque = [0.01, 0, 0, 0]

        # self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"{axisTorque}")

        self.wheelTorqueOutMsg.write(wheelTorqueOutMsgBuffer, CurrentSimNanos, self.moduleID)
        self.axisTorqueOutMsg.write(axisTorqueOutMsgBuffer, CurrentSimNanos, self.moduleID)
        self.attErrorOutMsg.write(attErrorOutMsgBuffer, CurrentSimNanos, self.moduleID)

        return


class SimManager(sysModel.SysModel):
    def __init__(self):
        super(SimManager, self).__init__()
        self.extTorqueOutMsg = messaging.CmdTorqueBodyMsg()
        self.rwAvaiOutMsg = messaging.RWAvailabilityMsg()

    def Reset(self, CurrentSimNanos):
        return

    def UpdateState(self, CurrentSimNanos):
        extTorqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()
        rwAvaiOutMsgBuffer = messaging.RWAvailabilityMsgPayload()

        if CurrentSimNanos < macros.sec2nano(10.0):
            extTorqueOutMsgBuffer.torqueRequestBody = [0, 0, 0]
        else:
            extTorqueOutMsgBuffer.torqueRequestBody = [0, 0, 0]

        if CurrentSimNanos < macros.sec2nano(1.0):
            rwAvaiOutMsgBuffer.wheelAvailability = [messaging.AVAILABLE, messaging.AVAILABLE, messaging.AVAILABLE, messaging.AVAILABLE]
        else:
            rwAvaiOutMsgBuffer.wheelAvailability = [messaging.AVAILABLE, messaging.AVAILABLE, messaging.AVAILABLE, messaging.AVAILABLE]

        self.extTorqueOutMsg.write(extTorqueOutMsgBuffer, CurrentSimNanos, self.moduleID)
        self.rwAvaiOutMsg.write(rwAvaiOutMsgBuffer, CurrentSimNanos, self.moduleID)
        return


if __name__ == "__main__":
    # reference, simulation time, timestep
    DynSpacecraft([-0.1, 0.5, -0.6], 20, 0.1)
    DynSpacecraft([0.5, 0.3, 0.7], 20, 0.1)
