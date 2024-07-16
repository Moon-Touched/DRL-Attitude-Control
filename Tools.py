import numpy as np


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
    print(phi, theta, psi)
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
