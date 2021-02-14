# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:03:22 2021

@author: Ounaye
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class LearnByMiddle(BaseEstimator, ClassifierMixin):
    
    # Code found at : https://scikit-learn.org/stable/developers/develop.html?fbclid=IwAR3oTTLl_HjIZxfJOsQabMN2D09KQGhfpuSqxK-2bCd1ZkyKedw6lqDSShE
    def __init__(self, demo_param='demo', merClass_ = 0, nonMerClass_ = 0):
        self.demo_param = demo_param
        self.merClass_ = merClass_
        self.nonMerClass_ = nonMerClass_

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        
        a,b = LearnByMiddle.determineCentreGrav(X, y)
        self.merClass_ = a
        self.nonMerClass_ = b
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        guess = np.zeros(len(X))
        
        for i in range(len(X)):
            guess[i] = LearnByMiddle.prendDecision(self.merClass_,self.nonMerClass_,X[i])
        
        # Dans le cas où on veut juste que ça marche remplacer le return
        # par la ligne en dessous
        # return guess
        
        # les lignes que j'ai suprimer par rapport à l'exemple : 
        # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        # return self.y_[guess]
        
        return self.y_[guess]
    
    # My code

    def determineCentreGrav(tabOfData,tabOfResult):
        centerOfTrue = np.zeros(len(tabOfData[0]))#Care
        nbrOfTrue = 0
        centerOfFalse = np.zeros(len(tabOfData[0]))
        nbrOfFalse = 0
        #Problème avec les nombres négatifs
        for i in range(len(tabOfData)):
            if (tabOfResult[i] == 1):
                nbrOfTrue += 1
                centerOfTrue += tabOfData[i]
            else:
                nbrOfFalse +=1
                centerOfFalse += tabOfData[i]
                
        return (centerOfTrue/nbrOfTrue,centerOfFalse/nbrOfFalse)


    def prendDecision(centreTrue,centreFalse,oneData):
        dist1 = np.linalg.norm(centreTrue-oneData)
        dist2 = np.linalg.norm(centreFalse-oneData)
        if dist1 > dist2:
            return -1
        else:
            return 1
            
                 
         
        