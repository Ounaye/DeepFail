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
        
# Conclusion : analyser les couleurs les présentes ça sert à pas grand chose

#Erreur dans le jeu de donné : Mer_3 pillq

# Après avoir regarder les images je remarques que détecter les images
# où il y a le plus de pixel bleu fait sens mais les programmes précédents
# ne font pas ça

# # Le problème de histogramme ( de ce que je comprend)
# c'est que on a pas vraiment une vision de pixel par pixel,
# il compte directement le nombre de pixel c'est pas pratique'

# Une liste interessante des autres manières de traiter une image
# https://moncoachdata.com/blog/10-outils-de-traitement-dimages-en-python/

import skimage




import glob as glb
from skimage import io
list_files = glb.glob("Data/Ailleurs/**/*.jpeg", recursive=True)


image_list = []
for filename in list_files:
        image_list.append(io.imread(filename))

# Donc ça donne une liste de numpy array, un par image 






# Lien utilisé pour récupérer les images https://docs.python.org/fr/3.6/library/glob.html

# import glob as glb
from numpy import zeros

import numpy as np

allImageNonMer = glb.glob("Data/Ailleurs/**/*.jpeg", recursive=True)
allImageMer = glb.glob("Data/Mer/**/*.jpeg", recursive=True)


tabOfData = zeros((len(allImageMer)+len(allImageNonMer),3))
tabOfResult = np.arange((len(allImageMer)+len(allImageNonMer)))

def makeTabForSk():
    index = 0
    for i in allImageMer: #1 à 4 secondes d'exécution
        tabOfData[index] = processImgByMaxSomme(i) # Modifier cette fonction
        tabOfResult[index] = 1
        index +=1
    for i in allImageNonMer:
        tabOfData[index] = processImgByMaxSomme(i)# Modifier cette fonction
        tabOfResult[index] = -1
        index +=1

print(tabOfData.shape)

makeTabForSk()

# A partir d'ici j'ai betement copier coller le TP pour faire marcher le truc
    
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(tabOfData, tabOfResult, test_size=0.20) 

from sklearn.naive_bayes import GaussianNB

classifieur = GaussianNB()

classifieur.fit(X_train, y_train)

y_predits = classifieur.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predits))




