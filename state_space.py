import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

T = 0.01
A = np.diag([0, 0, 0, 1, 1, 1])
B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
A_ZOH = linalg.expm(A * T)
B_ZOH = np.linalg.inv(A) @ (A_ZOH - np.eye(len(A))) @ B
u = np.array([[0.01], [0.01], [0.01]])

x_k=np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
x_history = [x_k]
for k in range(1, 6000):
    x_k = A_ZOH @ x_k + B_ZOH @ u
    x_history.append(x_k)

x_history=np.array(x_history)
plt.plot(x_history[:,0:3])
plt.show()