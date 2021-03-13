# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:23:59 2021

@author: Ounaye
"""

import glob as glb
from skimage import io

from joblib import  load
import skimage.transform as sky_trfm

import basicImageTraitement as BIT
import numpy as np
from skimage.feature import hog

import sys

def predict_main(rep):
    clf2 = load("clfStack.joblib")

    # Get the file
    list_filesNM = glb.glob(str(rep)+"/**", recursive=True)
    list_filesNM.pop(0)
    image_listNM = []
    for filename in list_filesNM:
        image_listNM.append(io.imread(filename))
    Xt = image_listNM

    # Machine learning stuff
    Xt = prepareTabForLearning(Xt)
    Yt = clf2.predict(Xt)
    Nt = list_filesNM
    print(Yt)
    return [Nt,Yt]

def prepareTabForLearning(tabOfImage):
    tabOfData = np.zeros((len(tabOfImage),140))
    index = 0
    for i in tabOfImage:
        print(index)
        otherPara = np.zeros(12)
        otherPara = BIT.littleColorHisto(i)
        shape = np.shape(i)
        fd, hog_image = hog(i, orientations=8, pixels_per_cell=(shape[0]/4, shape[1]/4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
        tabOfData[index] = np.concatenate((otherPara,fd))
        index +=1
    return tabOfData

def main():
    print("Hello World!")

if __name__ == "__main__":
    predict_main(sys.argv[1])
