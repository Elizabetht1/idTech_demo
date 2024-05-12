from matplotlib import pyplot as plt
import cv2 as cv ##open CV 
import numpy as np

def display(img): 
    fig,ax = plt.subplots()
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax.set_axis_off() 
    plt.show()

def removeNoise(img, seperate = True, iter = 1,kernel =3):
    k = np.ones((kernel, kernel), np.uint8)
    erode = cv.erode(img, k, iterations=iter) 
    dilate = cv.dilate(erode, k, iterations=iter)  
  
        
    if seperate:
        cv.imwrite("./results/erode.jpg",erode)
        cv.imwrite("./results/dilate.jpg",dilate)
    else: 
        cv.imwrite("./results/denoise_iter"+str(iter) +"_ksize"+str(kernel)+".jpg",dilate)
    return dilate

def runDenoise(img):
    removeNoise(img)
    removeNoise(img,kernel = 3,seperate = False)
    removeNoise(img,kernel = 5,seperate = False)
    removeNoise(img,kernel = 7,seperate = False)
    removeNoise(img,iter = 2,seperate = False)
    removeNoise(img,iter = 4,seperate = False)
    removeNoise(img,iter = 8,seperate = False)
