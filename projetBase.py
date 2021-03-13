# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:35:44 2021
@author: Ounaye
"""

"""
This is the main file of our project. It act as a main file.

We first open our image then we extract all the information
of our data. After this extraction we have 2 classification method

First a VotingClassifier, the weight he used is determined 
empirically. 

In Second we have a Stacking Classifier
"""

import basicImageTraitement as BIT
from numpy import zeros
import numpy as np
import fileManager
from skimage.feature import hog


import LearnByMiddle as LinearClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import  VotingClassifier
from sklearn.ensemble import  StackingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from joblib import dump

import sklearn.svm  as skSVM


import testAndFeedBack as testFunc

image_listNM, image_listM = fileManager.makeTabOfImg()


def prepareTabForLearning(tabOfM,tabOfNM,threshold):
    tabOfData = zeros((len(tabOfM)+len(tabOfNM),140))
    tabOfResult = np.arange((len(tabOfM)+len(tabOfNM)))
    index = 0
    for i in tabOfM:
        otherPara = zeros(12)
        otherPara = BIT.littleColorHisto(i)
        shape = np.shape(i)
        fd, hog_image = hog(i, orientations=8, pixels_per_cell=(shape[0]/4, shape[1]/4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
        tabOfData[index] = np.concatenate((otherPara,fd))
        tabOfResult[index] = 1
        index +=1
    for i in tabOfNM:
        otherPara = zeros(12)
        otherPara = BIT.littleColorHisto(i)
        shape = np.shape(i)
        fd, hog_image = hog(i, orientations=8, pixels_per_cell=(shape[0]/4, shape[1]/4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
        tabOfData[index] = np.concatenate((otherPara,fd))
        tabOfResult[index] = 0
        index +=1
    return (tabOfData,tabOfResult)


def makeTraining(tabData,tabResult): #Use to train one classifier alone
    
     
    X_train, X_test, y_train, y_test = train_test_split(tabData, tabResult, test_size=0.20) 
    
    
    classifieur = GaussianNB()
    classifieur.fit(X_train, y_train)
    
   
    y_predits = classifieur.predict(X_test)
    return accuracy_score(y_test,y_predits)



def makeTrainingWithVoting(tabData,tabResult):
    clf1 = Pipeline([('scaler', StandardScaler()), ('gauss',GaussianNB())])
    clf2 = Pipeline([('scaler', StandardScaler()), ('svm',skSVM.SVC(kernel ="linear"))])
    clf3 = Pipeline([('scaler', StandardScaler()), ('dtc',DecisionTreeClassifier(random_state=0,max_depth=2))])
    clf4 = Pipeline([('scaler', StandardScaler()), ('lc',LinearClassifier.LearnByMiddle())])
    
    X_train, X_test, y_train, y_test = train_test_split(tabData, tabResult, test_size=0.20)
    
    eclf = VotingClassifier(estimators = [("gnb", clf1), ("svm", clf2), ("dtc", clf3),("lc",clf4)],
                            weights=[0.15,0.24,0.14,0.20], voting='hard')
    eclf.fit(X_train, y_train)
    dump(eclf,"clfVote.joblib")
    
    return X_test,y_test

def makeTrainWithStakClassifier(tabData,tabResult):
    clf1 = Pipeline([('scaler', StandardScaler()), ('gauss',GaussianNB())])
    clf2 = Pipeline([('scaler', StandardScaler()), ('svm',skSVM.SVC(kernel ="linear"))])
    clf3 = Pipeline([('scaler', StandardScaler()), ('dtc',DecisionTreeClassifier(random_state=0,max_depth=2))])
    clf4 = Pipeline([('scaler', StandardScaler()), ('lc',LinearClassifier.LearnByMiddle())])
    
    
    X_train, X_test, y_train, y_test = train_test_split(tabData, tabResult, test_size=0.20)
    
    eclf = StackingClassifier(estimators = [("gnb", clf1), ("svm", clf2), ("dtc", clf3),("lc",clf4)])
    eclf.fit(X_train, y_train)
    dump(eclf,"clfStack.joblib")
    
    return X_test,y_test


"""
Function used to find the best Threshold for the BIT.analyseImg 
analyse function


"""
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



"""
How to make a Quick

a,b = prepareTabForLearning(image_listM,image_listNM, 1750)
testFunc.TestAndStats(a,b)

                
"""

