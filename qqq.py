import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, unitTestSupport, RigidBodyKinematics

print(RigidBodyKinematics.subMRP(np.array([0, 0, 0]), -np.array([1, 0, 0])).tolist())
print(RigidBodyKinematics.MRP2Euler123([0.1, 0.1, 0.1]) - RigidBodyKinematics.MRP2Euler123([0.2, 0.1, 0.1]))

print([1, 1, 1])
print(RigidBodyKinematics.euler1232MRP([1, 1, 1]))
print(np.array(np.array([1, 1, 1])))

e = []
e.append(([1, 1, 1], 10))
e.append(([1, 2, 1], 20))
df = pd.DataFrame(e)
df.to_csv("q.csv", index=False)
