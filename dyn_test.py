from Envs import BasiliskModel
from Basilisk.utilities import RigidBodyKinematics, macros
import numpy as np
import matplotlib.pyplot as plt


def PDControllor(error, omega):
    Kp = 0.1
    Kd = 0.1
    torque = -Kp * error - Kd * omega
    return torque


def random_euler():
    phi = np.random.uniform(-np.pi, np.pi)
    theta = np.random.uniform(-np.pi, np.pi)
    psi = np.random.uniform(-np.pi, np.pi)
    return np.array([phi, theta, psi])


if __name__ == "__main__":
    ref_MRP = RigidBodyKinematics.euler1232MRP(random_euler())
    # ref_MRP=np.array([0.5,0.5,0.5])
    print(ref_MRP)
    b_model = BasiliskModel(timestep=0.01, I=np.array([0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1]), ref_MRP=ref_MRP,torque_mode="axis")

    b_MRP_history = []
    b_MRP_history.append(b_model.cur_MRP)
    b_omega_history = []
    b_omega_history.append(b_model.cur_omega)
    b_error_MRP_history = []
    b_error_MRP_history.append(b_model.cur_error_MRP)
    for i in range(5000):

        torque = PDControllor(b_model.cur_error_MRP, b_model.cur_omega)
        cur_MRP, cur_error_MRP, cur_error_angle, cur_omega, cur_omega_dot = b_model.step(macros.sec2nano((i + 1) * 0.01), torque)
        b_MRP_history.append(cur_MRP)
        b_omega_history.append(cur_omega)
        b_error_MRP_history.append(cur_error_MRP)

    fig, axes = plt.subplots(2, 2)

    ax = axes[0, 0]
    ax.plot(b_MRP_history, label=["x", "y", "z"])
    ax.set_title("quat History")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(b_error_MRP_history, label=["error_x", "error_y", "error_z"])
    ax.set_title("Error History")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(b_omega_history, label=["omega_x", "omega_y", "omega_z"])
    ax.set_title("Angular Velocity History")
    ax.legend()

    ax = axes[1, 1]
    error_angle_history = []
    for mrp in b_error_MRP_history:
        error_angle_history.append(4 * np.arctan(np.linalg.norm(mrp)))
    ax.plot(error_angle_history, label="error")
    ax.set_title("Error History")
    ax.legend()
    plt.show()
