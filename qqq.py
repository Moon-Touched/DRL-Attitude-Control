import numpy as np


a=[ 0.42963459,  0.39684232, -0.42919071, -0.68827729]
map = np.array([[1, 0, 0, np.sqrt(3)/3], [0, 1, 0, np.sqrt(3)/3], [0, 0, 1, np.sqrt(3)/3]])
print(np.linalg.norm(a))
