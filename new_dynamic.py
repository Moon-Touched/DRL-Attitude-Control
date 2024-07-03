# rigidbody.py
# 四阶龙格库塔法建立刚体的运动学和动力学模型
import numpy as np
import matplotlib.pyplot as plt

# I: 转动惯量矩阵
class DynamicModel:
    def __init__(self, T_s: float, ref_quat: np.ndarray[np.float32], mode: int):
        self.T_s = T_s
        self.ref_quat = ref_quat
        self.mode = mode
        self.t = 0
        if mode == 1:
            self.map_matrix = np.array([[1, 0, 0, np.sqrt(3) / 3], [0, 1, 0, np.sqrt(3) / 3], [0, 0, 1, np.sqrt(3) / 3]])

        self.I = np.diag([0.025, 0.05, 0.065])
        self.I_inv = np.linalg.inv(self.I)

        self.pre_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.pre_omega = np.array([0.0, 0.0, 0.0])
        self.omega = np.array([0.0, 0.0, 0.0])

        self.pre_error = self.quat_error(self.pre_quat, self.ref_quat)
        self.pre_error = self.pre_error / np.linalg.norm(self.pre_error)
        self.pre_error = self.pre_error[1:]
        self.error = np.array([0.0, 0.0, 0.0])

        self.quat_history = [self.pre_quat]
        self.omega_history = [self.pre_omega]
        self.error_history = [self.pre_error]
        self.torque_history = [[0, 0, 0]]

    def Simulate_OneStep(self, torque_in: np.ndarray[np.float32]):
        if self.mode == 1:
            torque = np.dot(self.map_matrix, torque_in)
        else:
            torque = torque_in
        state = np.hstack([self.quat, self.omega])
        K1 = self._ODE4Function(self.t, state, torque)
        K2 = self._ODE4Function(self.t + self.T_s / 2, state + self.T_s / 2 * K1, torque)
        K3 = self._ODE4Function(self.t + self.T_s / 2, state + self.T_s / 2 * K2, torque)
        K4 = self._ODE4Function(self.t + self.T_s, state + self.T_s * K3, torque)
        diff = self.T_s / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        state += diff

        self.quat = state[0:4]
        self.omega = state[4:7]
        self.error = self.quat_error(self.quat, self.ref_quat)
        self.error = self.error / np.linalg.norm(self.error)
        self.error = self.error[1:]
        self.quat_history.append(self.quat)
        self.omega_history.append(self.omega)
        self.error_history.append(self.error)
        self.torque_history.append(torque)
        self.t += self.T_s

    def _ODE4Function(self, t, state, torque):
        Qs, Qv, omega = state[0], state[1:4], state[4:7]
        dQs = -0.5 * np.dot(Qv, omega)
        dQv = 0.5 * (Qs * omega + np.cross(Qv, omega))
        domega = self.I_inv @ (torque - np.cross(omega, self.I @ omega))
        return np.concatenate((np.array([dQs]), dQv, np.array(domega)))

    def quat_mul(self, p, q):
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

    def quat_error(self, q_cur, q_des):
        q_cur_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]]) / np.linalg.norm(q_cur)
        return self.quat_mul(q_des, q_cur_inv)


def eulerXYZ_to_quat(euler):
    c1 = np.cos(euler[0] / 2)
    s1 = np.sin(euler[0] / 2)
    c2 = np.cos(euler[1] / 2)
    s2 = np.sin(euler[1] / 2)
    c3 = np.cos(euler[2] / 2)
    s3 = np.sin(euler[2] / 2)

    q = np.array([c1 * c2 * c3 + s1 * s2 * s3, s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3])
    return q / np.linalg.norm(q)


def random_ref():
    psi = np.random.uniform(-np.pi, np.pi)  # yaw，偏航角，绕z轴
    theta = np.random.uniform(-np.pi / 2, np.pi / 2)  # pitch，俯仰角，绕y轴
    phi = np.random.uniform(-np.pi, np.pi)  # roll，横滚角，绕x轴
    print(phi, theta, psi)
    ref = eulerXYZ_to_quat(np.array([phi, theta, psi]))
    ref = ref / np.linalg.norm(ref)
    return ref


model = DynamicModel(0.01, [ 0.15640333,  0.29313238, -0.23092233, -0.9144869 ], 0)
Kp = 0.08
Kd = 0.5
for i in range(6000):
    torque = Kp * model.error - Kd * model.omega
    model.Simulate_OneStep(torque)
    
quat_history, omega_history, error_history, torque_history = model.quat_history, model.omega_history, model.error_history, model.torque_history
fig, axes = plt.subplots(2, 2)
ax = axes[0, 0]
ax.plot(quat_history, label=["w","x", "y", "z"])
ax.set_title("Quat History")
ax.legend()

ax = axes[0, 1]
ax.plot(error_history, label=["error_x", "error_y", "error_z"])
ax.set_title("Error History")
ax.legend()

ax = axes[1, 0]
ax.plot(omega_history, label=["omega_x", "omega_y", "omega_z"])
ax.set_title("Angular Velocity History")
ax.legend()

ax = axes[1, 1]
ax.plot(torque_history, label=["x", "y", "z"])
ax.set_title("Torque History")
ax.legend()
plt.show()