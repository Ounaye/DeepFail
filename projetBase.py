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

def analyseAllImg(threshold,tabImg):
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


list_filesNM = glb.glob("Data/Ailleurs/**/*.jpeg", recursive=True)
image_listNM = []
for filename in list_filesNM:
        image_listNM.append(io.imread(filename))

list_filesM = glb.glob("Data/Mer/**/*.jpeg", recursive=True)
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



def prepareTabForLearning(tabOfM,tabOfNM):
    tabOfData = zeros((len(tabOfM)+len(tabOfNM),3))
    tabOfResult = np.arange((len(tabOfM)+len(tabOfNM)))
    index = 0
    for i in tabOfM: #1 à 4 secondes d'exécution
        tabOfData[index] = analyseAllImg(2000,i)# Modifier cette fonction
        tabOfResult[index] = 1
        index +=1
    for i in tabOfNM:
        tabOfData[index] = analyseAllImg(2000,i)# Modifier cette fonction
        tabOfResult[index] = -1
        index +=1
    return (tabOfData,tabOfResult)
    


# A partir d'ici j'ai betement copier coller le TP pour faire marcher le truc


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def makeTraining(tabData,tabResult):
    
    #On fait nos samples test/entrainement    
    X_train, X_test, y_train, y_test = train_test_split(tabData, tabResult, test_size=0.20) 
    
    #On entraine
    classifieur = GaussianNB()
    classifieur.fit(X_train, y_train)
    
    #On test
    y_predits = classifieur.predict(X_test)
    print(accuracy_score(y_test,y_predits))


a,b = prepareTabForLearning(image_listM,image_listNM)
makeTraining(a,b)




