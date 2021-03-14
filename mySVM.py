# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 08:29:59 2021

@author: Ounaye
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle

class mySVM(BaseEstimator, ClassifierMixin):
    
    
    def __init__(self,maxIteration_ = 2050,learningRate_ = 0.000001,
                 reg_strenght_ = 10000):
         print("mySVM")
         self.maxIteration_ = maxIteration_
         self.learningRate_ = learningRate_
         self.reg_strenght_ = reg_strenght_
         
         
         

    
    def costFunction(self,weight,x,y): # On l'utilise pour évaluer l'avancé du modèle
        N = x.shape[0]
        dist = 1 - y *( np.dot(x,weight))
        dist[dist < 0] = 0
        cost = 0.5* np.dot(weight,weight) + (self.reg_strenght_ * (np.sum(dist)/N))
        return cost

    
    def gradiantCost(self,weight,xBatch,yBatch):
        
        if(type(yBatch) ==  np.float64 or type(yBatch) == np.int32 or type(yBatch) == np.int64):
            xBatch = np.array([xBatch])
            yBatch = np.array([yBatch])
            
        dst = 1 - (yBatch *np.dot(xBatch,weight)) # On regarde la distance
        downWeight = np.zeros(len(weight))
        index = 0
        
        if(type(dst) ==  np.float64):
            dst = np.array([dst])
        for i in dst:
            downI = 0
            if(max(0,i) == 0): # En dehors de notre marge on change rien
                downI = weight
            else: # Dans notre marge on ajuste le poids
                downI = weight - (self.reg_strenght_ * yBatch[index] * xBatch[index] )
            downWeight += downI
            index += 1
            
        downWeight = downWeight/len(dst) #Moyenne
        return downWeight


    def fit(self,features, outputs):
        tmp = np.ones((len(features),1))
        features = np.concatenate((features,tmp),axis=1)
        weights = np.zeros(features.shape[1])
        
        
        for epoch in range(self.maxIteration_):
            X, Y = shuffle(features, outputs)
            for ind, x in enumerate(X):
                ascent = self.gradiantCost(weights, x, Y[ind])
                weights = weights - (self.learningRate_ * ascent)
    

        self.weightVect_ = weights
        return self
    
    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        tmp = np.ones((len(X),1))
        X = np.concatenate((X,tmp),axis=1)
        
        guess = np.zeros(len(X),dtype=int)
        index = 0
        for i in X:
            tmp = np.dot(self.weightVect_,i)
            if( tmp > 0):
                guess[index] = 1
            elif(tmp < 0):
                guess[index] = -1
            else:
                guess[index] = 0
            index += 1
        
        #peut-etre modifier y_[guess]

        return guess

    
