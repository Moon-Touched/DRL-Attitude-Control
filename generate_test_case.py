import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = [-45, 0, 45]
targetAtt = []
for x in a:
    for y in a:
        for z in a:
            if not (x == 0 and y == 0 and z == 0):
                targetAtt.append([x, y, z])
print(targetAtt)

df = pd.DataFrame(targetAtt)
df.to_csv("testcase.csv", index=False)
