# rigidbody.py
# 四阶龙格库塔法建立刚体的运动学和动力学模型
import numpy as np
from quaternions import *

# 默认控制律，没有期望姿态
def Control_Law(state):
    Qv, W = state[1:4], state[4:7]
    return -np.array([2, 20, 6])*Qv -np.array([2, 20, 6])*W

# 刚体
# J: 刚体的转动惯量矩阵
# initEuler: 刚体的初始欧拉角
# initOmega: 刚体的初始角速度向量
class RigidBody:
    def __init__(self, J, initEuler, initOmega):
        self._J = J
        e1 = initEuler
        w1 = initOmega
        self._t = 0
        self._Jinv = np.linalg.inv(self._J)
        q1 = Euler_To_Quaternion(e1)
        self._states = np.concatenate((q1, w1), dtype=float)
        self.ctrlLaw = Control_Law  # 默认控制律

    def Simulate_OneStep(self):
        h = 0.01
        K1 = self._ODE4Function(self._t, self._states)
        K2 = self._ODE4Function(self._t+h/2, self._states + h/2*K1)
        K3 = self._ODE4Function(self._t+h/2, self._states + h/2*K2)
        K4 = self._ODE4Function(self._t+h, self._states + h*K3)
        dx = h/6*(K1 + 2*K2 + 2*K3 + K4)
        self._states += dx

    def Get_State(self):
        return self._states
    def Get_QuaternionVec(self):
        return Quaternion_to_Euler(self._states[0:4])

    def _ODE4Function(self, t, x):
        Qs, Qv, W = x[0], x[1:4], x[4:7]
        dQs = -0.5*np.dot(Qv, W)
        dQv = 0.5*(Qs*W + np.cross(Qv, W))
        torque = self.ctrlLaw(x)
        dW = self._Jinv @ (torque - np.cross(W, self._J @ W))[0]
        return np.concatenate((np.array([dQs]), dQv, np.array(dW)[0]))
