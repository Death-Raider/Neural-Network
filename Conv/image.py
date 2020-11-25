from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import json,sys
# with open("cost.txt") as File:
#     Cost = np.array(json.loads(File.read()))
#     def moving_average(x, w):
#         return np.convolve(x, np.ones(w), 'valid') / w
#     Cost = moving_average(Cost,int(len(Cost)/2))
#     print(Cost,Cost.shape)
#     plt.plot(Cost)
#     plt.show()
def read_in():
    value = sys.stdin.readlines()[0]
    return json.loads(value)
ConvData = read_in()
for i in ConvData:
    out = np.array(ConvData[i][0], dtype=np.float64)
    imgSave = Image.new("RGB",(len(out[0]),len(out)),"black")
    imgSave_value = imgSave.load()
    for x in range(imgSave.width):
        for y in range(imgSave.height):
            imgSave_value[x,y] = (int(out[y][x]*255),int(out[y][x]*255),int(out[y][x]*255))
    imgSave.save(f"images/image_{i}.png")
