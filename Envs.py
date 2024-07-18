import numpy as np
from Basilisk import __path__
from Basilisk.simulation import spacecraft, extForceTorque
from Basilisk.utilities import RigidBodyKinematics, unitTestSupport, macros, simIncludeRW, SimulationBaseClass
import gymnasium as gym
from gymnasium import spaces
import Tools, os

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


class BasiliskModel:
    def __init__(self, I: list, ref_MRP: np.ndarray, torque_mode: str = "wheel") -> None:
        self.ref_MRP = ref_MRP
        self.torque_mode = torque_mode
        if torque_mode == "wheel":
            self.map_matrix = np.array(
                [
                    [1, 0, 0, np.sqrt(3) / 3],
                    [0, 1, 0, np.sqrt(3) / 3],
                    [0, 0, 1, np.sqrt(3) / 3],
                ]
            )
        #  spacecraft  properties
        self.scObject = spacecraft.Spacecraft()
        self.scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)
        self.scObject.hub.sigma_BNInit = np.zeros(3, dtype=np.float32)
        self.scObject.hub.omega_BN_BInit = np.zeros(3, dtype=np.float32)
        self.extFTObject = extForceTorque.ExtForceTorque()
        self.extFTObject.extTorquePntB_B = [[0.0], [0.0], [0.0]]
        self.scObject.addDynamicEffector(self.extFTObject)

        self.scObject.SelfInit()
        # self.scObject.Reset()
        self.scObject.initializeDynamics()

        self.cur_MRP = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.cur_error_MRP = RigidBodyKinematics.subMRP(self.ref_MRP, self.cur_MRP)  # Q = subMRP(Q1,Q2) from Q2 to Q1.
        self.cur_error_angle = 4 * np.arctan(np.linalg.norm(self.cur_error_MRP))
        self.cur_omega = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.cur_omega_dot = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.MRP_history = [self.cur_MRP]
        self.error_MRP_history = [self.cur_error_MRP]
        self.error_angle_history = [self.cur_error_angle]
        self.omega_history = [self.cur_omega]
        self.omega_dot_history = [self.cur_omega_dot]

    def step(self, cur_nano_second: int, torque: np.ndarray):
        if self.torque_mode == "wheel":
            self.extFTObject.extTorquePntB_B = np.dot(self.map_matrix, torque)
        elif self.torque_mode == "axis":
            self.extFTObject.extTorquePntB_B = torque
        else:
            raise ValueError("torque_mode must be 'wheel' or 'axis'")

        self.scObject.UpdateState(cur_nano_second)

        state = self.scObject.scStateOutMsg.read()
        self.cur_MRP = np.array(state.sigma_BN, dtype=np.float32)
        self.cur_error_MRP = RigidBodyKinematics.subMRP(self.ref_MRP, self.cur_MRP)
        self.cur_error_angle = 4 * np.arctan(np.linalg.norm(self.cur_error_MRP))
        self.cur_omega = np.array(state.omega_BN_B, dtype=np.float32)
        self.cur_omega_dot = np.array(state.omegaDot_BN_B, dtype=np.float32)

        self.MRP_history.append(self.cur_MRP)
        self.error_MRP_history.append(self.cur_error_MRP)
        self.error_angle_history.append(self.cur_error_angle)
        self.omega_history.append(self.cur_omega)
        self.omega_dot_history.append(self.cur_omega_dot)

        return self.cur_MRP, self.cur_error_MRP, self.cur_error_angle, self.cur_omega, self.cur_omega_dot


class BasiliskEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console", faulty: bool = False, torque_mode: str = "wheel"):
        super(BasiliskEnv, self).__init__()
        self.faulty = faulty
        self.torque_mode = torque_mode
        self.render_mode = render_mode
        self.model: BasiliskModel = None
        self.ref_MRP = np.array([0.0, 0.0, 0.0])
        self.reward = 0
        self.action = np.zeros(4, dtype=np.float32)  # 四轮力矩
        self.observation = np.zeros(6, dtype=np.float32)  # 角度误差，角速度
        self.step_count = 0
        self.fault_time = 0
        self.wheel_num = 0
        if self.torque_mode == "wheel":
            self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        elif self.torque_mode == "axis":
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            raise ValueError("torque_mode must be 'wheel' or 'axis'")
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.ref_MRP = RigidBodyKinematics.euler1232MRP(Tools.random_euler())
        self.model = BasiliskModel(I=[0.025, 0, 0, 0, 0.05, 0, 0, 0, 0.065], ref_MRP=self.ref_MRP, torque_mode=self.torque_mode)
        self.reward = 0
        self.action = np.zeros(4, dtype=np.float32)  # 四轮力矩
        self.observation = np.zeros(6, dtype=np.float32)  # 误差MRP，角速度
        self.step_count = 0
        self.fault_time = np.random.randint(low=0, high=3000)
        self.wheel_num = np.random.randint(low=-1, high=4)

        self.observation = np.hstack([self.model.cur_error_MRP, self.model.cur_omega], dtype=np.float32)

        # if self.faulty:
        #     print(f"reference:{self.ref_MRP}")
        #     print(f"faultTime:{self.fault_time}")
        #     print(f"wheelNum:{self.wheel_num}")
        # else:
        #     print("no fault")
        #     print(f"reference:{self.ref_MRP}")
        return self.observation, {}

    def step(self, action):
        terminated = False
        truncated = False
        self.step_count += 1
        self.action = 0.01 * action  # gym推荐action范围[-1,1]，再进行缩放

        # 设置故障轮力矩为0
        if self.faulty:
            if self.step_count > self.fault_time and self.wheel_num != -1:
                self.action[self.wheel_num] = 0

        self.model.step(macros.sec2nano((self.step_count) * 0.01), self.action)

        self.observation = np.hstack([self.model.cur_error_MRP, self.model.cur_omega], dtype=np.float32)

        self.reward = self.calculate_reward()

        if self.step_count == 6000:
            truncated = True
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            terminated = True

        info = {}
        return self.observation, self.reward, terminated, truncated, info

    def calculate_reward(self):

        return

    def render(self):
        pass

    def close(self):
        pass
