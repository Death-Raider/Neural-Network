from PIL import Image
import json
def createImg(img):
    with open("images/"+img+".txt") as file:
        file = json.loads(file.read())
        imgSave = Image.new("RGB",(len(file[0]), len(file)),"black")
        imgSave_value = imgSave.load()
        for x in range(imgSave.width):
            for y in range(imgSave.height):
                imgSave_value[x,y] = (int(file[y][x]*255),int(file[y][x]*255),int(file[y][x]*255))
        imgSave.save("images/"+img+".png")
        print(imgSave.width,imgSave.height)
        imgSave.close()
createImg("image")
createImg("convo1")
createImg("convo2")
