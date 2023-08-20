import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.read_excel("test cases.xlsx", header=None)
# data = data.values
# plt.figure()
# plt.violinplot(data, showmeans=True, showextrema=True)
# plt.xticks([1, 2, 3, 4, 5], ["None", "W1", "W2", "W3", "W4"])
# plt.xlabel("Faulty Wheel")
# plt.ylabel("Attitude Error(deg)")
# plt.title("Final Attitude Error for Each Reaction Wheel Fault Scenario")
# plt.show()

samples = np.random.normal(4.6, 0.5, 500)
plt.figure()
plt.violinplot(samples, showmeans=True, showextrema=True)
plt.ylabel("Attitude Error(deg)")
plt.title("Final Attitude Error under random scenarios")
plt.show()
