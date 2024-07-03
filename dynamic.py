import numpy as np
import matplotlib.pyplot as plt


class DynamicModel:
    # mode: 0： 直接输入三轴力矩, 1： 输入四轮转矩再映射到三轴
    # 四轮分布为[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]

    def __init__(self, T_s: float, ref_quat: np.ndarray[np.float32], mode: int):
        self.T_s = T_s
        self.ref_quat = ref_quat
        self.mode = mode
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

    def step(self, torque_in: np.ndarray[np.float32]):
        if self.mode == 1:
            torque = np.dot(self.map_matrix, torque_in)
        else:
            torque = torque_in
        self.torque_history.append(torque)
        omega_dot = self.I_inv @ (torque - np.cross(self.pre_omega, np.dot(self.I, self.pre_omega)))
        self.omega = self.pre_omega + omega_dot * self.T_s
        self.omega_history.append(self.omega)

        theta = (self.omega + self.pre_omega) / 2 * self.T_s
        theta_norm = np.linalg.norm(theta)
        if theta_norm > 0:  # 避免除以零
            qw = np.cos(theta_norm / 2)
            qv = theta / theta_norm * np.sin(theta_norm / 2)
            q_trans = np.array([qw, qv[0], qv[1], qv[2]])
        else:
            q_trans = np.array([1.0, 0.0, 0.0, 0.0])

        self.quat = self.quat_mul(self.pre_quat, q_trans)
        self.quat = self.quat / np.linalg.norm(self.quat)
        self.quat_history.append(self.quat)

        self.error = self.quat_error(self.quat, self.ref_quat)
        self.error = self.error / np.linalg.norm(self.error)
        self.error = self.error[1:]
        self.error_history.append(self.error)

        self.pre_quat, self.pre_omega, self.pre_error = self.quat, self.omega, self.error

        return self.quat, self.omega, self.error

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

    def quat_to_eulerXYZ(self, quat):
        qw, qx, qy, qz = quat
        phi = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        theta = np.arcsin(2 * (qw * qy - qx * qz))
        psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([phi, theta, psi])


def PDtest(mode):
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

    steps = 6000
    T_s = 0.01
    Kp = 0.08
    Kd = 0.5
    model = DynamicModel(T_s, random_ref(), mode)
    print(model.ref_quat)
    quat, omega, error = model.pre_quat, model.pre_omega, model.pre_error
    for i in range(steps):
        torque = Kp * error - Kd * omega
        quat, omega, error = model.step(torque)
    return model.quat_history, model.omega_history, model.error_history, model.torque_history


if __name__ == "__main__":

    def quat_to_eulerXYZ(quat):
        qw, qx, qy, qz = quat
        phi = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        theta = np.arcsin(2 * (qw * qy - qx * qz))
        psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([phi, theta, psi])

    quat_history, omega_history, error_history, torque_history = PDtest(0)
    fig, axes = plt.subplots(2, 2)
    rpy_history = []
    for q in quat_history:
        rpy_history.append(quat_to_eulerXYZ(q))

    ax = axes[0, 0]
    ax.plot(quat_history, label=["w", "x", "y", "z"])
    ax.set_title("quat History")
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
