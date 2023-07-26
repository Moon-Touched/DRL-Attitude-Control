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


def DynSpacecraft_noWheel(attref, totalTime, timestep):
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
    scSim.AddModelToTask(simTaskName, scObject, None, 10)

    # external torque
    extTorque = extForceTorque.ExtForceTorque()
    extTorque.ModelTag = "externalTorque"
    scObject.addDynamicEffector(extTorque)
    scSim.AddModelToTask(simTaskName, extTorque)

    extTorqueOutMsg = messaging.CmdTorqueBodyMsg()
    extTorqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()
    extTorqueOutMsgBuffer.torqueRequestBody = [0.1, 0, 0]
    extTorqueOutMsg.write(extTorqueOutMsgBuffer)

    # link messages
    extTorque.cmdTorqueInMsg.subscribeTo(extTorqueOutMsg)

    stateLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, stateLog)

    # cmdTorqueLog = controller.axisTorqueOutMsg.recorder()
    # scSim.AddModelToTask(simTaskName, cmdTorqueLog)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ShowExecutionOrder()
    scSim.ExecuteSimulation()

    timeAxis = stateLog.times()
    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.sigma_BN[:, idx], color=unitTestSupport.getLineColor(idx, 3), label="omega" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Current MRP")

    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.omega_BN_B[:, idx], color=unitTestSupport.getLineColor(idx, 3), label="omega" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Current angular velocity [r/s]")

    plt.show()

    return


def DynSpacecraft_3Wheel(attref, totalTime, timestep):
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
    scSim.AddModelToTask(simTaskName, scObject, None, 10)

    rwFactory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels
    RW1 = rwFactory.create("Honeywell_HR16", [1, 0, 0], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
    RW2 = rwFactory.create("Honeywell_HR16", [0, 1, 0], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
    RW3 = rwFactory.create("Honeywell_HR16", [0, 0, 1], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)

    numRW = rwFactory.getNumOfDevices()

    #  RW  container
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwStateEffector.ModelTag = "RW_cluster"
    rwFactory.addToSpacecraft(scObject.ModelTag, rwStateEffector, scObject)
    scSim.AddModelToTask(simTaskName, rwStateEffector, None, 20)

    # external torque
    extTorque = extForceTorque.ExtForceTorque()
    extTorque.ModelTag = "externalTorque"
    scObject.addDynamicEffector(extTorque)
    scSim.AddModelToTask(simTaskName, extTorque, None, 1)

    extTorqueOutMsg = messaging.CmdTorqueBodyMsg()
    extTorqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()
    extTorqueOutMsgBuffer.torqueRequestBody = [0.1, 0, 0]
    extTorqueOutMsg.write(extTorqueOutMsgBuffer)

    # link messages
    extTorque.cmdTorqueInMsg.subscribeTo(extTorqueOutMsg)

    stateLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, stateLog)

    # cmdTorqueLog = controller.axisTorqueOutMsg.recorder()
    # scSim.AddModelToTask(simTaskName, cmdTorqueLog)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ShowExecutionOrder()
    scSim.ExecuteSimulation()

    timeAxis = stateLog.times()
    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.sigma_BN[:, idx], color=unitTestSupport.getLineColor(idx, 3), label="omega" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Current MRP")

    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.omega_BN_B[:, idx], color=unitTestSupport.getLineColor(idx, 3), label="omega" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Current angular velocity [r/s]")

    plt.show()

    return


def DynSpacecraft_4Wheel(attref, totalTime, timestep):
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
    scSim.AddModelToTask(simTaskName, scObject, None, 10)

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

    # external torque
    extTorque = extForceTorque.ExtForceTorque()
    extTorque.ModelTag = "externalTorque"
    scObject.addDynamicEffector(extTorque)
    scSim.AddModelToTask(simTaskName, extTorque, None, 1)

    extTorqueOutMsg = messaging.CmdTorqueBodyMsg()
    extTorqueOutMsgBuffer = messaging.CmdTorqueBodyMsgPayload()
    extTorqueOutMsgBuffer.torqueRequestBody = [0.1, 0, 0]
    extTorqueOutMsg.write(extTorqueOutMsgBuffer)

    # link messages
    extTorque.cmdTorqueInMsg.subscribeTo(extTorqueOutMsg)

    stateLog = scObject.scStateOutMsg.recorder()
    scSim.AddModelToTask(simTaskName, stateLog)

    # cmdTorqueLog = controller.axisTorqueOutMsg.recorder()
    # scSim.AddModelToTask(simTaskName, cmdTorqueLog)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(simulationTime)
    scSim.ShowExecutionOrder()
    scSim.ExecuteSimulation()

    timeAxis = stateLog.times()
    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.sigma_BN[:, idx], color=unitTestSupport.getLineColor(idx, 3), label="omega" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Current MRP")

    plt.figure()
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2SEC, stateLog.omega_BN_B[:, idx], color=unitTestSupport.getLineColor(idx, 3), label="omega" + str(idx + 1))
    plt.legend(loc="lower right")
    plt.xlabel("Time [s]")
    plt.ylabel("Current angular velocity [r/s]")

    plt.show()

    return


if __name__ == "__main__":
    # reference, simulation time, timestep
    DynSpacecraft_4Wheel([0.0, 0.0, 0.0], 1, 0.1)
