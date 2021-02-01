# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:35:44 2021
@author: Ounaye
"""

from PIL import Image 
from matplotlib import pyplot as plt


  
def openImg(str):
    img = Image.open(str)
    r,g,b = img.split()
    return r, g, b

# MaxColor : Dumb Test 0,5 de prédiction

def findMax(histo):
    tmp = 0
    for i in histo:
        if tmp < i:
            tmp = i
    return tmp

def processImgByMax(str):
    r,g,b = openImg(str)
    maxR = findMax(r.histogram())
    maxG = findMax(g.histogram())
    maxB = findMax(b.histogram())
    tmp = maxR
    if(tmp < maxG):
        tmp = maxG
    if(tmp < maxB):
        tmp = maxB
    return tmp

# SommeMaxColor : StillDumb 0,5 de prédiction

def findMaxSomme(histo):
    tmp = 0
    for i in histo:
        tmp += i
    return i

def processImgByMaxSomme(str):
    r,g,b = openImg(str)
    maxR = findMaxSomme(r.histogram())
    maxG = findMaxSomme(g.histogram())
    maxB = findMaxSomme(b.histogram())
    return [maxR,maxG,maxB]

# On essaye de faire un traitement sur plus de pixel
# 0.6 de réussite

def isBlue(pixel):
    blueValue = pixel[2]
    if(pixel[0] < blueValue and pixel[1] < blueValue):
        return True
    return False

def analyseImageBlue(imageArray):
    nbrBlue = 0
    indexI = -1
    indexJ = -1
    for i in imageArray:
        for j in imageArray[indexI]:
            if(isBlue(imageArray[indexI,indexJ])):
                nbrBlue += 1
            indexJ +=1
        indexI += 1
        indexJ = 0
    return nbrBlue

def analyseImg(threshold,tabImg):
    if(analyseImageBlue(tabImg) > threshold):
        return 1
    else:
        return -1





import glob as glb
from numpy import zeros
from skimage import io
import skimage.transform as sky_trfm
import numpy as np


# Si on enlève le filtrage .jpeg 
# Il a besoin de format de lecture en arg


list_filesNM = glb.glob("DataNonRéduite/Ailleurs/**/*.jpeg", recursive=True)
image_listNM = []
for filename in list_filesNM:
        image_listNM.append(io.imread(filename))

list_filesM = glb.glob("DataNonRéduite/Mer/**/*.jpeg", recursive=True)
image_listM = []
for filename in list_filesM:
        image_listM.append(io.imread(filename))



def rescaleImg(listImg,listNameImg):
#Flemme de gérer le warning pour l'instant
    for i in range(len(listImg)):
        tmpImg = sky_trfm.resize(listImg[i], (64,64))
        io.imsave(listNameImg[i], tmpImg)

# Modifie totalement les images !!
# rescaleImg(image_listNM, list_filesNM)
#rescaleImg(image_listM,list_filesM)



def prepareTabForLearning(tabOfM,tabOfNM,threshold):
    tabOfData = zeros((len(tabOfM)+len(tabOfNM),3))
    tabOfResult = np.arange((len(tabOfM)+len(tabOfNM)))
    index = 0
    for i in tabOfM: #1 à 4 secondes d'exécution
        grad,sobX,sobY = detectForme(i)
        tabOfData[index,0] = analyseImg(threshold,i)# Modifier cette fonction
        tabOfData[index,1] = sobX
        tabOfResult[index] = 1
        index +=1
    for i in tabOfNM:
        tabOfData[index] = analyseImg(threshold,i)# Modifier cette fonction
        tabOfResult[index] = -1
        index +=1
    return (tabOfData,tabOfResult)

    
#Détection de contours  

#image réduite
import cv2 as cv
import time


def detectForme(img):
    img = cv.imread(img,0)
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

data = tab

plt.hist(data)

plt.title('How to plot a simple histogram in matplotlib ?', fontsize=10)

plt.savefig("plot_simple_histogramme_matplotlib_01.png")

plt.show()





# A partir d'ici j'ai betement copier coller le TP pour faire marcher le truc


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def makeTraining(tabData,tabResult):
    
    # On fait nos samples test/entrainement    
    X_train, X_test, y_train, y_test = train_test_split(tabData, tabResult, test_size=0.20) 
    
    # On entraine
    classifieur = GaussianNB()
    classifieur.fit(X_train, y_train)
    
    # On test
    y_predits = classifieur.predict(X_test)
    return accuracy_score(y_test,y_predits)
    
    
def findBestThreshold(debut, step):
    maxTreshold = -800000
    maxScore = -800000
    for i in range (debut,4001, step):
        score = 0
        for j in range(10):
            a,b = prepareTabForLearning(image_listM,image_listNM, i)
            score += makeTraining(a,b)
        if (score >= maxScore) :
            maxScore = score
            maxTreshold = i

    return maxTreshold


# a,b = prepareTabForLearning(image_listM,image_listNM, 1750)
# makeTraining(a, b)
            
