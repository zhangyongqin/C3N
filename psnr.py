import os
import numpy as np
import math
import cv2
import time
start = time.clock()

def psnr(img1, img2):
    mse = np.mean((img1/1. - img2/1.) ** 2 )
    if mse < 1.0e-10:
        return 100*1.0
    return 10 * math.log10(255.0*255.0/mse)

def mse(img1,img2):
    mse = np.mean((img1/1. - img2/1.) ** 2 )
    return mse
    
 
def ssim(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

path2 = '/team_stor1/huayu/pc-3/val_256/' 
path1 = '/team_stor1/huayu/pc-3/result/'
f_nums = len(os.listdir(path1))
list_psnr = []
list_ssim = []
list_mse = []
for i in range(1,f_nums+1):
    img_a = cv2.imread(path1+"result"+str(i)+'.jpg')
    img_b = cv2.imread(path2+str(i)+'.jpg')
    psnr_num = psnr(img_a, img_b)
    ssim_num = ssim(img_a, img_b)
    mse_num = mse(img_a,img_b)
    
    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)
    list_mse.append(mse_num)
    
    
print("PSNR:",np.mean(list_psnr))

print("SSIM:",np.mean(list_ssim))

print("MSE:",np.mean(list_mse))


 

elapsed = (time.clock() - start)

print("Time used:",elapsed)
