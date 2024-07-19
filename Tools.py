import numpy as np
import matplotlib.pyplot as plt
from Envs import BasiliskEnv, MyEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv


class BEnv1(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if self.step_count == 6000 and cur_error < 0.0043633:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class BEnv2(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (10 - cur_error) * (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if cur_error < 0.0043633 and np.linalg.norm(self.model.cur_omega) < 0.001:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class BEnv3(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 1 / (cur_error + 1)

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class BEnv4(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / pre_error

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if cur_error < 0.0043633:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class BEnv5(BasiliskEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if cur_error < 0.0043633:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class MyEnv1(MyEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if self.step_count == 6000 and cur_error < 0.0043633:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class MyEnv2(MyEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (10 - cur_error) * (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if cur_error < 0.0043633 and np.linalg.norm(self.model.cur_omega) < 0.001:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class MyEnv3(MyEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / np.pi

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 1 / (cur_error + 1)

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


class MyEnv4(MyEnv):
    def calculate_reward(self):
        pre_error = self.model.error_angle_history[-2]
        cur_error = self.model.error_angle_history[-1]

        r1 = (pre_error - cur_error) / pre_error

        r2 = 0
        if np.abs(self.model.cur_omega[0]) > 1 or np.abs(self.model.cur_omega[1]) > 1 or np.abs(self.model.cur_omega[2]) > 1:
            r2 = -1

        r3 = 0
        if cur_error < 0.0043633:
            r3 = 1

        reward = r1 + r2 + r3
        # print(f"r1: {r1}, r2: {r2}")
        return reward


def eulerXYZ_to_quat(euler: np.ndarray) -> np.ndarray:
    c1 = np.cos(euler[0] / 2)
    s1 = np.sin(euler[0] / 2)
    c2 = np.cos(euler[1] / 2)
    s2 = np.sin(euler[1] / 2)
    c3 = np.cos(euler[2] / 2)
    s3 = np.sin(euler[2] / 2)

    q = np.array([c1 * c2 * c3 + s1 * s2 * s3, s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3])
    return q / np.linalg.norm(q)


def random_quar_ref() -> np.ndarray:
    psi = np.random.uniform(-np.pi, np.pi)  # yaw，偏航角，绕z轴
    theta = np.random.uniform(-np.pi / 2, np.pi / 2)  # pitch，俯仰角，绕y轴
    phi = np.random.uniform(-np.pi, np.pi)  # roll，横滚角，绕x轴
    # print(phi, theta, psi)
    ref = eulerXYZ_to_quat(np.array([phi, theta, psi]))
    ref = ref / np.linalg.norm(ref)
    return ref


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_error(q_cur: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    q_cur_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]]) / np.linalg.norm(q_cur)
    q_cur_inv = q_cur_inv / np.linalg.norm(q_cur_inv)
    return quat_mul(q_des, q_cur_inv)


def mrp_to_quaternion(mrp: np.ndarray) -> np.ndarray:
    mrp_squared_norm = np.dot(mrp, mrp)
    scalar_part = (1 - mrp_squared_norm) / (1 + mrp_squared_norm)
    vector_part = (2 * mrp) / (1 + mrp_squared_norm)
    quaternion = np.hstack((scalar_part, vector_part))

    return quaternion


def random_euler():
    phi = np.random.uniform(-np.pi, np.pi)
    theta = np.random.uniform(-np.pi, np.pi)
    psi = np.random.uniform(-np.pi, np.pi)

    return np.array([phi, theta, psi])


def plot_history(dynamic_model):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax = axes[0, 0]
    ax.plot(dynamic_model.MRP_history, label=["x", "y", "z"])
    ax.set_title("MRP History")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(dynamic_model.error_MRP_history, label=["error_x", "error_y", "error_z"])
    ax.set_title("Error History")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(dynamic_model.omega_history, label=["omega_x", "omega_y", "omega_z"])
    ax.set_title("Angular Velocity History")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(dynamic_model.error_angle_history, label="error")
    ax.set_title("error_angle_history")
    ax.legend()
    return fig, axes


def make_env(env_name: str, env_num: int, faulty: bool, torque_mode: str, vec_env_cls=DummyVecEnv):
    env_classes = {
        "benv01": BEnv1,
        "benv02": BEnv2,
        "benv03": BEnv3,
        "benv04": BEnv4,
        "benv05": BEnv5,
        "menv01": MyEnv1,
        "menv02": MyEnv2,
        "menv03": MyEnv3,
        "menv04": MyEnv4,
    }
    if env_name not in env_classes.keys():
        raise ValueError("env name not found")

    if env_num == 1:
        env = env_classes[env_name](faulty=faulty, torque_mode=torque_mode)
        env.reset()
    else:
        env = make_vec_env(
            env_classes[env_name], n_envs=env_num, env_kwargs={"faulty": faulty, "torque_mode": torque_mode}, vec_env_cls=vec_env_cls
        )

    return env
