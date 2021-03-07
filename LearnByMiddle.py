# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:03:22 2021
@author: Ounaye
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class LearnByMiddle(BaseEstimator, ClassifierMixin):
    
    # Code found at : https://scikit-learn.org/stable/developers/develop.html?fbclid=IwAR3oTTLl_HjIZxfJOsQabMN2D09KQGhfpuSqxK-2bCd1ZkyKedw6lqDSShE
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        
        a,b = LearnByMiddle.determineCentreGrav(X, y)
        self.merMiddle_ = a
        self.nonMerMiddle_ = b
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self


#prend en entre un classifieur et un ensemble de donnée X
#prédire les étiquettes de X
    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        
        guess = np.zeros(len(X),dtype=int)
        
        
        #peut-etre modifier y_[guess]
        for i in range(len(X)):
            #pour debugger, afficher la taille de self.merClass
            guess[i] = LearnByMiddle.prendDecision(self.merMiddle_,self.nonMerMiddle_,X[i])
        
        return guess

    
    # My code

    def determineCentreGrav(tabOfData,tabOfResult):
        centerOfTrue = np.zeros(len(tabOfData[0]))
        nbrOfTrue = 0
        centerOfFalse = np.zeros(len(tabOfData[0]))
        nbrOfFalse = 0
        #Problème avec les nombres négatifs
        for i in range(len(tabOfData)):
            if (tabOfResult[i] == 1):
                nbrOfTrue += 1.0
                centerOfTrue += tabOfData[i]
            else:
                nbrOfFalse +=1.0
                centerOfFalse += tabOfData[i]
                
        return (centerOfTrue/nbrOfTrue,centerOfFalse/nbrOfFalse)

# s'assurer que centreTrue, centreFalse et oneData ont la même taille
    def prendDecision(centreTrue,centreFalse,oneData):
        dist1 = np.linalg.norm(centreTrue-oneData)
        dist2 = np.linalg.norm(centreFalse-oneData)
        if dist1 > dist2:
            return 0
        else:
            return 1

