# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:54:40 2021

@author: Ounaye
"""

"""
We make some basic statistics to analyse our result and plot some graph
"""

import matplotlib.pyplot as plt
import math
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


from joblib import load

def testOnData(tabDataTest,tabResultTest,stack = True ):
    
    X_train, X_test, y_train, y_test = train_test_split(tabDataTest, tabResultTest, test_size=0.20)
    if(stack):
        eclf = load("clfStack.joblib")
    else:
        eclf = load("clfVote.joblib")
    y_predits = eclf.predict(X_test)
    accM,accNM=getMoreResult(y_predits,y_test)
    return accuracy_score(y_test,y_predits),accM,accNM

def testOnDataQuick(tabDataTest,tabResultTest,stack = True ):
    
    X_train, X_test, y_train, y_test = train_test_split(tabDataTest, tabResultTest, test_size=0.20)
    if(stack):
        eclf = load("clfStack.joblib")
    else:
        eclf = load("clfVote.joblib")
    y_predits = eclf.predict(X_test)

    return accuracy_score(y_test,y_predits)


def TestAndStats(tabDataTest,tabResultTest):
    tabOfTrain = np.zeros(100)
    axisX = np.zeros(100)
    avg = 0
    avgAccM = 0
    avgAccNM = 0
    for i in range(100):
        tabOfTrain[i],accM,accNM = testOnData(tabDataTest,tabResultTest,stack=False)
        axisX[i] = i
        avg+=tabOfTrain[i]
        avgAccM += accM
        avgAccNM += accNM
    plt.plot(axisX,tabOfTrain)
    print("Moyenne : " + str(avg/100))
    print("Ecart Type : " + str(math.sqrt(np.var(tabOfTrain, 0))))
    print("Accuracy Moyenne de image MER : " + str(avgAccM/100))
    print("Accuracy Moyenne de image NONMER : " + str(avgAccNM/100))


def TestAndStatsQuick(tabDataTest,tabResultTest):
    tabOfTrain = np.zeros(100)
    avg = 0
    for i in range(100):
        tabOfTrain[i]= testOnDataQuick(tabDataTest,tabResultTest)
        avg+=tabOfTrain[i]
    print("Moyenne : " + str(avg/100))
    print("Ecart Type : " + str(math.sqrt(np.var(tabOfTrain, 0))))


def getMoreResult(tabPredict,tabSol):
    nbrTT = 0
    nbrTF = 0
    nbrFT = 0
    nbrFF = 0
    for i in range(len(tabPredict)):
        if(tabSol[i] == 0): #NM
            if(tabPredict[i] == 0):
                nbrFF+=1
            else:
                nbrFT+=1
        else:
            if(tabPredict[i] == 0):
                nbrTF+=1
            else:
                nbrTT+=1
    accuracyMer = nbrTT/(nbrTT + nbrTF)
    accuracyNonMer = nbrFF/(nbrFT + nbrFF)
    return accuracyMer,accuracyNonMer
