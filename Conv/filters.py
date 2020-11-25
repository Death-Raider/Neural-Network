from PIL import Image
import numpy as np
import json,sys
def read_in():
    value = sys.stdin.readlines()[0]
    return json.loads(value)
Filters = read_in()
i = 0
for filter in Filters:
    filter = np.array(filter)
    filter = filter/filter.max()
    imgSave = Image.new("RGB",(len(filter),len(filter)),"black")
    imgSave_value = imgSave.load()
    for x in range(imgSave.width):
        for y in range(imgSave.height):
            imgSave_value[x,y] = (int(filter[y][x]*255),int(filter[y][x]*255),int(filter[y][x]*255))
    imgSave.save(f"Filters/filter{i}.png")
    i+=1

with open("Filters/Filters_BEFORE.txt") as File:
    Filters = np.array(json.loads(File.read()))
    i = 0
    for filter in Filters:
        filter = np.array(filter)
        imgSave = Image.new("RGB",(len(filter),len(filter)),"black")
        imgSave_value = imgSave.load()
        for x in range(imgSave.width):
            for y in range(imgSave.height):
                imgSave_value[x,y] = (int(filter[y][x]*255),int(filter[y][x]*255),int(filter[y][x]*255))
        imgSave.save(f"Filters/filter{i}_BEFORE.png")
        i+=1
