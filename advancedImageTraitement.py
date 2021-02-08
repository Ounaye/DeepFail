# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:44:11 2021

@author: Ounaye
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

#Détection de contours  

#image réduite

def detectForme(img):
    #img = cv.imread(img,0)
    img= cv.GaussianBlur(img, (3, 3), 0)
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    abs_grad_x = cv.convertScaleAbs(sobelx)
    abs_grad_y = cv.convertScaleAbs(sobely)
    modgrad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #On ajoute nos deux images l'une sur l'autre
    return modgrad,sobelx,sobely

def makeHistogramOfGrad(img,pas,nbrBarre):
    tabHistogramme = np.zeros(nbrBarre)
    for i in range(0,64,pas):
        for j in range(0,64,pas):
                analyseGradImg(img, tabHistogramme,nbrBarre,i,j,pas)
    return tabHistogramme
        
def analyseGradImg(img,tabHist,nbrBarre,xDebut,yDebut,pas):
    img= cv.GaussianBlur(img, (3, 3), 0)
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
    mag, angle = cv.cartToPolar(sobelx, sobely, angleInDegrees=True)
    if angle.shape == (64,64,3):
        return
    for i in range(xDebut,xDebut+pas):
        for j in range(yDebut,yDebut+pas):
            a = angle[i,j]/(180/nbrBarre)
            b = int(a) % 10
            if(a == b ):
                 tabHist[b] += mag[i,j]
            else:
                tabHist[b] += mag[i,j]*(a - b)
                tabHist[(b+1)%10] += mag[i,j]* (a - b + 1)
            


imgdaz = cv.imread("Data/Mer/Mer_1/aaaaa.jpeg",0)
tab  = makeHistogramOfGrad(imgdaz,8,10)
plt.bar([0,20,40,60,80,100,120,140,160,180],tab,width=19.0)
plt.show();

