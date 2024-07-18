import numpy as np

map_matrix = np.array([[1, 0, 0, np.sqrt(3) / 3], [0, 1, 0, np.sqrt(3) / 3], [0, 0, 1, np.sqrt(3) / 3]])
t = np.array([0.16666666666666657, 0.16666666666666657, -0.8333333333333329, -0.28867513459481275])
print(np.dot(map_matrix, t))


axis_torque = [0, 0, -1]
