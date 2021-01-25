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

from skimage import feature
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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



def prepareTabForLearning(tabOfM,tabOfNM,threshold):
    tabOfData = zeros((len(tabOfM)+len(tabOfNM),3))
    tabOfResult = np.arange((len(tabOfM)+len(tabOfNM)))
    index = 0
    for i in tabOfM: #1 à 4 secondes d'exécution
        tabOfData[index] = analyseAllImg(threshold,i)# Modifier cette fonction
        tabOfResult[index] = 1
        index +=1
    for i in tabOfNM:
        tabOfData[index] = analyseAllImg(threshold,i)# Modifier cette fonction
        tabOfResult[index] = -1
        index +=1
    return (tabOfData,tabOfResult)


def findOutLinesOfImage(pathName):
#---------- Read Image ----------#
    img = mpimg.imread(pathName)
    print (type(img))
    print (img.shape, img.dtype)
    print (img[63,63,0],img[63,63,1],img[63,63,2])
    print (img.max(),img.min())

    M = np.zeros((img.shape[0],img.shape[1]))
    print (M)
    M[:,:] = img[:,:,0]
    print (M.max(),M.min(),M.shape)
    plt.imshow(M, cmap = plt.get_cmap('gray'))
    plt.title("Lena Picture")
    plt.savefig("Lena.png")
    plt.show()
#---------- Apply Canny  ----------#
    edges = feature.canny(M, sigma=2)
    fig, ax = plt.subplots()
    ax.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
    #ax.axis('off')
    ax.set_title('Canny Edge Detection')
    plt.savefig("LenaCanny.png")
    plt.show()
    

def findOutLinesOfImage2(pathName):

    im1 = Image.open(pathName)
    im2 = Image.new("RGB",(64,64))
    im3 = Image.new("RGB",(64,64))
    for y in range(64):
        for x in range(64):
            p = im1.getpixel((x,y))        
            r = (p[0]+p[1]+p[2])/3
            v = r
            b = r
            im2.putpixel((x,y),(r,v,b))
    for y in range(0,63):
        print (y)
        for x in range(0,63):
            pix0 = im2.getpixel((x,y))
            pix1 = im2.getpixel((x-1,y-1))
            pix2 = im2.getpixel((x,y-1))
            pix3 = im2.getpixel((x+1,y-1))
            pix4 = im2.getpixel((x-1,y))
            pix5 = im2.getpixel((x+1,y))
            pix6 = im2.getpixel((x-1,y+1))
            pix7 = im2.getpixel((x,y+1))
            pix8 = im2.getpixel((x+1,y+1))
            r = 8*pix0[0]-pix1[0]-pix2[0]- pix3[0]-pix4[0]-pix5[0]-pix6[0]-pix7[0]-pix8[0]
        r = r/1
        r = r+128
        r = 255-r
        v = r
        b = r
    im3.putpixel((x,y),(r,v,b))
    im3.save("T:\Seville_contours.png")
    im3.show()

    


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
            



findOutLinesOfImage2("Data/Mer/Mer_1/aaaaa.jpeg")
        
        
#v = findBestThreshold(1000, 250)
#print(v)
#a,b = prepareTabForLearning(image_listM,image_listNM,1750)
#print(makeTraining(a, b))








