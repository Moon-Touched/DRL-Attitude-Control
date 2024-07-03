import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dynamic import DynamicModel


class MyEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console"):
        super(MyEnv, self).__init__()

        self.render_mode = render_mode
        # 以下在reset中都会重新赋值，此处只是声明便于查看
        self.model: DynamicModel = None
        self.reference = np.array([1.0, 0.0, 0.0, 0.0])
        self.reward = 0
        self.action = np.zeros(4)  # 四轮力矩
        self.observation = np.zeros(6)  # 角度误差，角速度
        self.stepCount = 0
        self.faultTime = 0
        self.wheelNum = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reference = self.random_ref()
        self.model: DynamicModel = DynamicModel(0.01, self.reference, mode=1)
        self.reward = 0
        self.action = np.zeros(4)  # 四轮力矩
        self.observation = np.zeros(10)  # 四元数，角度误差，角速度
        self.stepCount = 0
        self.faultTime = np.random.randint(low=0, high=3000)
        self.wheelNum = np.random.randint(low=-1, high=4)

        self.observation = np.hstack([self.model.pre_error, self.model.pre_omega])
        print(f"reference:{self.reference}")
        print(f"faultTime:{self.faultTime}")
        print(f"wheelNum:{self.wheelNum}")
        return self.observation, {}

    def step(self, action):
        terminated = False
        truncated = False
        self.stepCount = self.stepCount + 1
        self.action = 0.01 * action  # gym推荐的action范围[-1,1]保证性能，这里乘以0.01是为了限制力矩的大小
        # 设置故障轮力矩为0
        if self.stepCount > self.faultTime:
            if self.wheelNum != -1:
                self.action[self.wheelNum] = 0

        self.model.step(self.action)
        self.observation = np.hstack([self.model.pre_error, self.model.pre_omega])

        pre_error = 4 * np.arctan(np.linalg.norm(self.model.error_history[-2]))
        cur_error = 4 * np.arctan(np.linalg.norm(self.model.error_history[-1]))

        r1 = 0
        if self.stepCount == 6000 and cur_error < 0.0043633:
            r1 = 1

        r2 = (pre_error - cur_error) / np.pi

        r3 = 0
        if np.abs(self.model.pre_omega[0]) > 1 or np.abs(self.model.pre_omega[1]) > 1 or np.abs(self.model.pre_omega[2]) > 1:
            r3 = -1
            terminated = True

        self.reward = r1 + r2 + r3

        if self.stepCount > 6000:
            truncated = True
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        return self.observation, self.reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def eulerXYZ_to_quat(self, euler):
        c1 = np.cos(euler[0] / 2)
        s1 = np.sin(euler[0] / 2)
        c2 = np.cos(euler[1] / 2)
        s2 = np.sin(euler[1] / 2)
        c3 = np.cos(euler[2] / 2)
        s3 = np.sin(euler[2] / 2)

        q = np.array([c1 * c2 * c3 + s1 * s2 * s3, s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3])
        return q / np.linalg.norm(q)

    def random_ref(self):
        psi = np.random.uniform(-np.pi, np.pi)  # yaw，偏航角，绕z轴
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)  # pitch，俯仰角，绕y轴
        phi = np.random.uniform(-np.pi, np.pi)  # roll，横滚角，绕x轴
        ref = self.eulerXYZ_to_quat(np.array([phi, theta, psi]))
        ref = ref / np.linalg.norm(ref)
        return ref
