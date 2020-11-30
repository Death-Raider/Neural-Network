from matplotlib import pyplot as plt
import numpy as np
import json
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
with open("costC.txt") as File:
    CostC = np.array(json.loads(File.read()))
    CostC = moving_average(CostC,10)
    print(CostC,CostC.shape)
with open("costN.txt") as File:
    CostN = np.array(json.loads(File.read()))
    CostN = moving_average(CostN,10)
    print(CostN,CostN.shape)
plt.plot(CostC,label="CNN")
plt.plot(CostN,label="ANN")
plt.legend()
plt.show()
