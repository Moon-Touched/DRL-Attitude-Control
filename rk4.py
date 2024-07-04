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

        # 误差四元数，取虚部作为控制输入
        self.pre_error = self.quat_error(self.pre_quat, self.ref_quat)
        self.pre_error = self.pre_error / np.linalg.norm(self.pre_error)
        self.error = np.array([1.0, 0.0, 0.0, 0.0])

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

        self.pre_quat, self.pre_omega, self.pre_error = self.quat, self.omega, self.error

        return self.quat, self.omega, self.error

    def rk4(self, torque):
        def omega_dot(torque):
            return self.I_inv @ (torque - np.cross(self.pre_omega, np.dot(self.I, self.pre_omega)))

        k1_omega = omega_dot(torque)
        k1_q = self.quat_mul(self.pre_quat, np.hstack([0, self.pre_omega])) / 2

        # Compute k2
        omega_half = self.pre_omega + 0.5 * self.T_s * k1_omega
        q_half = self.pre_quat + 0.5 * self.T_s * k1_q
        k2_omega = omega_dot(omega_half, J, T)
        k2_q = q_dot(q_half, omega_half)

        # Compute k3
        omega_half = omega + 0.5 * dt * k2_omega
        q_half = q + 0.5 * dt * k2_q
        k3_omega = omega_dot(omega_half, J, T)
        k3_q = q_dot(q_half, omega_half)

        # Compute k4
        omega_full = omega + dt * k3_omega
        q_full = q + dt * k3_q
        k4_omega = omega_dot(omega_full, J, T)
        k4_q = q_dot(q_full, omega_full)

        # Update omega and q
        omega_next = omega + (dt / 6) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)
        q_next = q + (dt / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

        # Normalize quaternion
        q_next = q_next / np.linalg.norm(q_next)

        return omega_next, q_next

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

    def draw(self, error_format: str = "quat"):
        fig, axes = plt.subplots(2, 2)

        ax = axes[0, 0]
        ax.plot(self.quat_history, label=["w", "x", "y", "z"])
        ax.set_title("quat History")
        ax.legend()

        ax = axes[0, 1]
        if error_format == "quat":
            ax.plot(self.error_history, label=["error_w", "error_x", "error_y", "error_z"])
        elif error_format == "angle":
            e_angles = []
            for q in self.error_history:
                e_angles.append(2 * np.arccos(q[0]))
            ax.plot(e_angles, label=["error"])
        ax.set_title("Error History")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(self.omega_history, label=["omega_x", "omega_y", "omega_z"])
        ax.set_title("Angular Velocity History")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(self.torque_history, label=["x", "y", "z"])
        ax.set_title("Torque History")
        ax.legend()
        plt.show()
