import numpy as np

from PIL import Image

def L1(yhat, y):

    loss = np.sum(np.abs(y - yhat))

    return loss

def L2(yhat, y):

    loss =np.sum(np.power((y - yhat), 2))

    return loss

for i in range(1,2):

    imgA = Image.open("/team_stor1/huayu/pc-3/A/"+str(i)+".jpg")

    yhat = np.array(imgA)
    
    imgB = Image.open("/team_stor1/huayu/pc-3/B/"+str(i)+".jpg")

    y = np.array(imgB)

    print("L1 = " ,(L1(yhat,y)))
    print("L2 = " ,(L2(yhat,y)))