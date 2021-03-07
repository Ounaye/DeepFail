# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:35:28 2021

@author: Ounaye
"""


from PIL import Image


  
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
def isARed(pixel):
    if(pixel[0] > 120 and pixel[1] < 100):
        return True
    return False

def isAGreen(pixel):
    if(pixel[0] < 100 and pixel[1] > 120 and pixel[2] < 80):
        return True
    return False

def isABlue(pixel):
    if(pixel[0] < 50 and pixel[1] < 90 and pixel[2] > 120):
        return True
    return False


def analyseColorImg(imageArray):
    nbrBlue = 0
    nbrRed = 0
    nbrGreen = 0
    indexI = -1
    indexJ = -1
    for i in imageArray:
        for j in imageArray[indexI]:
            pixel = imageArray[indexI,indexJ]
            if(isARed(pixel)):
                nbrRed +=1
            if(isAGreen(pixel)):
                nbrGreen+=1
            if(isABlue(pixel)):
                nbrBlue+=1
            indexJ +=1
        indexI += 1
        indexJ = 0
    return nbrRed,nbrGreen,nbrGreen


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
