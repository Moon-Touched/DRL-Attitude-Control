import os

import matplotlib.pyplot as plt
import numpy as np
# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import (mrpFeedback, attTrackingError,
                                    inertial3D, rwMotorTorque, rwMotorVoltage)
from Basilisk.simulation import reactionWheelStateEffector, motorVoltageInterface, simpleNav, spacecraft, extForceTorque
from Basilisk.utilities import (SimulationBaseClass, fswSetupRW, macros,
                                orbitalMotion, simIncludeGravBody,
                                simIncludeRW, unitTestSupport, vizSupport)

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


def DynSpacecraft(attref):
    # Simulation Parameters
    simTaskName = "simTask"
    simProcessName = "simProcess"
    scSim = SimulationBaseClass.SimBaseClass()
    simulationTime = macros.sec2nano(30.)
    simulationTimeStep = macros.sec2nano(0.1)

    # create the simulation process and task
    dynProcess = scSim.CreateNewProcess(simProcessName)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    #  spacecraft  properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "mySpacecraft"
    I = [0.025, 0., 0.,
         0., 0.05, 0.,
         0., 0., 0.065]
    scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
    scSim.AddModelToTask(simTaskName, scObject, None, 1)

    # Reaction wheels
    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels
    RW1 = rwFactory.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=0., RWModel=varRWModel)
    RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=0., RWModel=varRWModel)
    RW3 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=0., RWModel=varRWModel)

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

    # external torque
    disturbance = Disturbance()
    disturbance.ModelTag = "Disturbance"
    scSim.AddModelToTask(simTaskName, disturbance)
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

    fswRwParamMsg = rwFactory.getConfigMessage()
    # link messages
    sNavObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)
    attErrorConfig.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
    attErrorConfig.attRefInMsg.subscribeTo(inertial3DConfig.attRefOutMsg)
    controller.stateInMsg.subscribeTo(scObject.scStateOutMsg)
    controller.mrpRefInMsg.subscribeTo(inertial3DConfig.attRefOutMsg)
    extTorque.cmdTorqueInMsg.subscribeTo(disturbance.cmdTorqueOutMsg)
    rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    rwMotorTorqueConfig.vehControlInMsg.subscribeTo(controller.cmdTorqueOutMsg)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueConfig.rwMotorTorqueOutMsg)

    # Log record
    stateLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, stateLog)
    rwMotorLog = rwMotorTorqueConfig.rwMotorTorqueOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, rwMotorLog)
    attErrorLog = attErrorConfig.attGuidOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, attErrorLog)
    disturbanceLog = disturbance.cmdTorqueOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, disturbanceLog)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ShowExecutionOrder()
    scSim.ExecuteSimulation()

    timeAxis = attErrorLog.times()
    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, attErrorLog.sigma_BR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='sigma' + str(idx))
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('MRP Error')

    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.sigma_BN[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='sigma' + str(idx))
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Current MRP')

    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.omegaDot_BN_B[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='omega_dot' + str(idx))
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('Current angular acceleration [r/s^2]')

    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, rwMotorLog.motorTorque[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label='T' + str(idx))
    plt.legend(loc='lower right')
    plt.xlabel('Time [s]')
    plt.ylabel('RW Torque')

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
        self.k = 1
        self.P = 1
        self.stateInMsg = messaging.SCStatesMsgReader()
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()
        self.mrpRefInMsg = messaging.AttRefMsgReader()

    def Reset(self, CurrentSimNanos):
        return

    def UpdateState(self, CurrentSimNanos):
        # read input message
        stateBuffer = self.stateInMsg()
        mrpRefBuffer = self.mrpRefInMsg()
        # create output message buffer
        torqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()
        torqueOutMsgBuffer.torqueRequestBody = -1 * (self.k * (mrpRefBuffer.sigma_RN - stateBuffer.sigma_BN))
        self.cmdTorqueOutMsg.write(torqueOutMsgBuffer, CurrentSimNanos, self.moduleID)
        return


class Disturbance(sysModel.SysModel):

    def __init__(self):
        super(Disturbance, self).__init__()

        # Output body torque message name
        self.cmdTorqueOutMsg = messaging.CmdTorqueBodyMsg()

    def Reset(self, CurrentSimNanos):
        return

    def UpdateState(self, CurrentSimNanos):

        # create output message buffer
        torqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()

        if CurrentSimNanos < macros.sec2nano(5.):
            torqueOutMsgBuffer.torqueRequestBody = [0, 0, 0]
        else:
            torqueOutMsgBuffer.torqueRequestBody = [0.1, 0, 0]
        self.cmdTorqueOutMsg.write(torqueOutMsgBuffer, CurrentSimNanos, self.moduleID)

        return


if __name__ == "__main__":
    DynSpacecraft([0.5, 0, 0])
