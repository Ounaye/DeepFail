# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:35:44 2021

@author: Ounaye
"""


# from PIL.Image import *
# from matplotlib import pyplot as plt


# img = open("Data/Ailleurs/Ailleurs3/tagt0o.jpeg")


# A=img.histogram()

# R=A[ 0:255 ]
# G=A[ 256:511 ]
# B=A[ 512:767 ]

# xs=range(255)
# plt.scatter(xs,R, c='red')
# plt.scatter(xs,G, c='green')
# plt.scatter(xs,B, c='blue')

from PIL import Image 
from matplotlib import pyplot as plt
  
# img = Image.open("Data/Ailleurs/Ailleurs3/tagt0o.jpeg") 
# r, g, b = img.split() 
# print(len(r.histogram())) 
# ### 256 ### 

# xs=range(256)
# plt.scatter(xs,r.histogram(), c='red')
# plt.scatter(xs,g.histogram(), c='green')
# plt.scatter(xs,b.histogram(), c='blue')
# 256*256*3 

# Variable : Une Matrice de (Matrice de taille 768) 
#On fait la max du bleu, la max du vert et la max du rouge
#On a donc 3 données par images et on voit ce que ça donne

# Resultat : Bool

  
def openImg(str):
    img = Image.open(str)
    r,g,b = img.split()
    return r, g, b

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




# Lien utilisé pour récupérer les images https://docs.python.org/fr/3.6/library/glob.html

import glob as glb
from numpy import zeros

import numpy as np

allImageNonMer = glb.glob("Data/Ailleurs/**/*.jpeg", recursive=True)
allImageMer = glb.glob("Data/Mer/**/*.jpeg", recursive=True)


tabOfData = zeros((len(allImageMer)+len(allImageNonMer),1))
tabOfResult = np.arange((len(allImageMer)+len(allImageNonMer)))

index = 0
for i in allImageMer: #1 à 4 secondes d'exécution
    tabOfData[index] = processImgByMax(i)
    tabOfResult[index] = 1
    index +=1
for i in allImageNonMer:
    tabOfData[index] = processImgByMax(i)
    tabOfResult[index] = -1
    index +=1

# A partir d'ici j'ai betement copier coller le TP pour faire marcher le truc


print(tabOfData.shape)

    
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(tabOfData, tabOfResult, test_size=0.20) 

from sklearn.naive_bayes import GaussianNB

classifieur = GaussianNB()

classifieur.fit(X_train, y_train)

y_predits = classifieur.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predits))




