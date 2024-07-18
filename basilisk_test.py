import os
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


class TestModel:
    def __init__(self, timestep: int, I: list, ref_MRP: list) -> None:
        self.scSim = SimulationBaseClass.SimBaseClass()
        task_name = "sim_task"
        process_name = "sim_process"
        dyn_process = self.scSim.CreateNewProcess(process_name)
        dyn_process.addTask(self.scSim.CreateNewTask(task_name, timestep))

        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "mySpacecraft"
        self.scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
        self.scObject.hub.sigma_BNInit = [0, 0, 0]
        self.scObject.hub.omega_BN_BInit = [0, 0, 0]
        self.scSim.AddModelToTask(task_name, self.scObject, None)

        # Reaction wheels
        rwFactory = simIncludeRW.rwFactory()
        varRWModel = messaging.BalancedWheels
        RW1 = rwFactory.create("Honeywell_HR16", [1, 0, 0], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
        RW2 = rwFactory.create("Honeywell_HR16", [0, 1, 0], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
        RW3 = rwFactory.create("Honeywell_HR16", [0, 0, 1], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel)
        RW4 = rwFactory.create(
            "Honeywell_HR16", [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3], maxMomentum=50.0, Omega=0.0, RWModel=varRWModel
        )
        numRW = rwFactory.getNumOfDevices()

        # RW  container
        self.rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        self.rwStateEffector.ModelTag = "RW_cluster"
        rwFactory.addToSpacecraft(self.scObject.ModelTag, self.rwStateEffector, self.scObject)
        self.scSim.AddModelToTask(task_name, self.rwStateEffector, None)
        rwParamMsg = rwFactory.getConfigMessage()

        # self.rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
        # self.rwMotorTorqueObj.ModelTag = "rwMotorTorque"
        # self.scSim.AddModelToTask(task_name, self.rwMotorTorqueObj)

        # # Make the RW control all three body axes
        # controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        # self.rwMotorTorqueObj.controlAxes_B = controlAxes_B

        # link messages
        # torque msg
        self.wheelTorqueMsg = messaging.ArrayMotorTorqueMsg()
        # self.wheelTorqueMsg = messaging.CmdTorqueBodyMsg()
        self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.wheelTorqueMsg)
        # self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.rwMotorTorqueObj.rwMotorTorqueOutMsg)
        # self.rwMotorTorqueObj.vehControlInMsg.subscribeTo(self.wheelTorqueMsg)
        # self.rwMotorTorqueObj.rwParamsInMsg.subscribeTo(rwParamMsg)
        self.scSim.InitializeSimulation()

        self.ref_MRP = ref_MRP
        self.cur_MRP = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.cur_error_MRP = RigidBodyKinematics.subMRP(self.ref_MRP, self.cur_MRP)  # Q = subMRP(Q1,Q2) from Q2 to Q1.
        self.cur_error_angle = 4 * np.arctan(np.linalg.norm(self.cur_MRP))
        self.cur_omega = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.cur_omega_dot = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.MRP_history = [self.cur_MRP]
        self.erroe_MRP_history = [self.cur_error_MRP]
        self.error_angle_history = [self.cur_error_angle]
        self.omega_history = [self.cur_omega]
        self.omega_dot_history = [self.cur_omega_dot]

    def step(self, stop_time, torque):
        wheelTorqueBuffer = messaging.ArrayMotorTorqueMsgPayload()
        wheelTorqueBuffer.motorTorque = torque
        # wheelTorqueBuffer = messaging.CmdTorqueBodyMsgPayload()
        # wheelTorqueBuffer.torqueRequestBody = torque
        self.wheelTorqueMsg.write(wheelTorqueBuffer)
        self.scSim.ConfigureStopTime(stop_time)
        self.scSim.ExecuteSimulation()

        state = self.scObject.scStateOutMsg.read()
        self.cur_MRP = np.array(state.sigma_BN, dtype=np.float32)
        self.cur_error_MRP = RigidBodyKinematics.subMRP(self.ref_MRP, self.cur_MRP)
        self.cur_error_angle = 4 * np.arctan(np.linalg.norm(self.cur_MRP))
        self.cur_omega = np.array(state.omega_BN_B, dtype=np.float32)
        self.cur_omega_dot = np.array(state.omegaDot_BN_B, dtype=np.float32)

        self.MRP_history.append(self.cur_MRP)
        self.erroe_MRP_history.append(self.cur_error_MRP)
        self.error_angle_history.append(self.cur_error_angle)
        self.omega_history.append(self.cur_omega)
        self.omega_dot_history.append(self.cur_omega_dot)
        # self.wheel_speed = self.rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds
        # self.wheel_torque=self.rwMotorTorqueObj.rwMotorTorqueOutMsg.read().motorTorque


if __name__ == "__main__":
    I = [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
    timestep = macros.sec2nano(0.01)
    m = TestModel(timestep, I, np.array([0.5, 0.5, 0.5], dtype=np.float32))
    for i in range(1000):
        stop_time = macros.sec2nano((i + 1) * 0.01)
        m.step(stop_time, [1, 0, 0, 0])
    print(m.omega_dot_history[-1])
    plt.plot(m.omega_dot_history)
    plt.show()
