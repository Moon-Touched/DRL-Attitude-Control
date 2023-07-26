import os

import matplotlib.pyplot as plt
import numpy as np

# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import attTrackingError, inertial3D, rwMotorTorque
from Basilisk.simulation import reactionWheelStateEffector, simpleNav, spacecraft, extForceTorque
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, unitTestSupport

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


def DynSpacecraft(attref, totalTime, timestep):
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
    scSim.AddModelToTask(simTaskName, scObject, None, 1)

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
    scSim.AddModelToTask(simTaskName, rwStateEffector, None, 2)

    # simple Navigation sensor
    sNavObject = simpleNav.SimpleNav()
    sNavObject.ModelTag = "SimpleNavigation"
    scSim.AddModelToTask(simTaskName, sNavObject)

    # setup inertial3D guidance module
    inertial3DConfig = inertial3D.inertial3DConfig()
    inertial3DWrap = scSim.setModelDataWrap(inertial3DConfig)
    inertial3DWrap.ModelTag = "inertial3D"
    scSim.AddModelToTask(simTaskName, inertial3DWrap, inertial3DConfig)
    inertial3DConfig.sigma_R0N = attref

    # setup the attitude tracking error evaluation module
    attErrorConfig = attTrackingError.attTrackingErrorConfig()
    attErrorWrap = scSim.setModelDataWrap(attErrorConfig)
    attErrorWrap.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attErrorWrap, attErrorConfig)

    # setup the controller
    controller = Controller()
    controller.ModelTag = "myController"
    scSim.AddModelToTask(simTaskName, controller)

    # Sim Manager
    simManager = SimManager()
    simManager.ModelTag = "SimManager"
    scSim.AddModelToTask(simTaskName, simManager)

    # external torque
    extTorque = extForceTorque.ExtForceTorque()
    extTorque.ModelTag = "externalTorque"
    scObject.addDynamicEffector(extTorque)
    scSim.AddModelToTask(simTaskName, extTorque)

    # add module that maps the Lr control torque into the RW motor torques
    rwMotorTorqueConfig = rwMotorTorque.rwMotorTorqueConfig()
    rwMotorTorqueWrap = scSim.setModelDataWrap(rwMotorTorqueConfig)
    rwMotorTorqueWrap.ModelTag = "rwMotorTorque"
    scSim.AddModelToTask(simTaskName, rwMotorTorqueWrap, rwMotorTorqueConfig)

    # Make the RW control all three body axes
    controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    rwMotorTorqueConfig.controlAxes_B = controlAxes_B

    rwParamMsg = rwFactory.getConfigMessage()

    # link messages
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attErrorConfig.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attErrorConfig.attRefInMsg.subscribeTo(inertial3DConfig.attRefOutMsg)
    controller.guidInMsg.subscribeTo(attErrorConfig.attGuidOutMsg)
    extTorque.cmdTorqueInMsg.subscribeTo(simManager.extTorqueOutMsg)
    rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(rwParamMsg)
    rwMotorTorqueConfig.vehControlInMsg.subscribeTo(controller.cmdTorqueOutMsg)
    rwMotorTorqueConfig.rwAvailInMsg.subscribeTo(simManager.rwAvaiOutMsg)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueConfig.rwMotorTorqueOutMsg)

    # Log record
    stateLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, stateLog)

    attErrorLog = attErrorConfig.attGuidOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, attErrorLog)

    rwMotorLog = rwMotorTorqueConfig.rwMotorTorqueOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, rwMotorLog)

    cmdTorqueLog = controller.cmdTorqueOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, cmdTorqueLog)

    disturbanceLog = simManager.extTorqueOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, disturbanceLog)

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

    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, cmdTorqueLog.torqueRequestBody[:, idx], label="T" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Axis Torque")

    # plt.figure()
    # for idx in range(3):
    #     plt.plot(timeAxis * macros.NANO2SEC, disturbanceLog.torqueRequestBody[:, idx],
    #              color=unitTestSupport.getLineColor(idx, 3),
    #              label='T' + str(idx))
    # plt.legend(loc='lower right')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Disturbance Torque')

    plt.show()

    return


class Controller(sysModel.SysModel):
    def __init__(self):
        super(Controller, self).__init__()
        self.Kp = 0.1
        self.Kd = 0.1
        # Input guidance structure message
        self.guidInMsg = messaging.AttGuidMsgReader()
        # Output body torque message name
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

    def Reset(self, CurrentSimNanos):
        return

    def UpdateState(self, CurrentSimNanos):
        guidMsgBuffer = self.guidInMsg()
        torqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()

        # compute control solution
        lrCmd = np.array(guidMsgBuffer.sigma_BR) * self.Kp + np.array(guidMsgBuffer.omega_BR_B) * self.Kd
        # lrCmd = np.array([0, 1, 0])
        torqueOutMsgBuffer.torqueRequestBody = (lrCmd).tolist()

        self.cmdTorqueOutMsg.write(torqueOutMsgBuffer, CurrentSimNanos, self.moduleID)
        # self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"Time: {CurrentSimNanos * 1.0E-9} s")
        # self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"TorqueRequestBody: {torqueOutMsgBuffer.torqueRequestBody}")
        # self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"sigma_BR: {guidMsgBuffer.sigma_BR}")
        # self.bskLogger.bskLog(sysModel.BSK_INFORMATION, f"omega_BR_B: {guidMsgBuffer.omega_BR_B}")
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

        if CurrentSimNanos < macros.sec2nano(2.0):
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
    DynSpacecraft([0.5, 0.2, 0.7], 20, 0.1)
