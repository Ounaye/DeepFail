# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:35:44 2021
@author: Ounaye
"""


import basicImageTraitement as BIT
from numpy import zeros
import numpy as np
import fileManager
import advancedImageTraitement as AIT

image_listNM, image_listM = fileManager.makeTabOfImg()


def prepareTabForLearning(tabOfM,tabOfNM,threshold):
    tabOfData = zeros((len(tabOfM)+len(tabOfNM),11))
    tabOfResult = np.arange((len(tabOfM)+len(tabOfNM)))
    index = 0
    for i in tabOfM: #1 à 4 secondes d'exécution
        hist = AIT.makeHistogramOfGrad(i,8,10)
        tabOfData[index,0] = BIT.analyseImg(threshold,i)
        for i in range(10):
            tabOfData[index,i+1] = hist[i]
        tabOfResult[index] = 1
        index +=1
    for i in tabOfNM:
        tabOfData[index,0] = BIT.analyseImg(threshold,i)
        for i in range(10):
            tabOfData[index,i+1] = hist[i]
        tabOfResult[index] = 1
        tabOfResult[index] = -1
        index +=1
    return (tabOfData,tabOfResult)

    




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


a,b = prepareTabForLearning(image_listM,image_listNM, 1750)
makeTraining(a, b)
            
