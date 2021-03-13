# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:39:54 2021

@author: Ounaye
"""

"""
This file is use to handle every file related task

We have here :
    a method to get our image in the dataset
    a method to rescale an image to 64*64
    a method to make copy of our dataset but with slight modification
    it's used to make the dataset bigger and reduced the noise.
    Also this method rescale our image by 64*60. All the data passed to
    our programs must be of 64*60 !

"""

import glob as glb
from skimage import io
import skimage.transform as sky_trfm

import transformImage as trfImg
import skimage.util as sk_util


def makeTabOfImg():
    list_filesNM = glb.glob("DataEnhanced/Ailleurs/**/*.jpeg", recursive=True)
    image_listNM = []
    for filename in list_filesNM:
        image_listNM.append(io.imread(filename))
    
    list_filesM = glb.glob("DataEnhanced/Mer/**/*.jpeg", recursive=True)
    image_listM = []
    for filename in list_filesM:
        image_listM.append(io.imread(filename))
        
    return (image_listNM,image_listM)

"""
All the method below produce a warning, we didn't have the time to 
handle that.
"""


def rescaleImg64(listImg,listNameImg):
    for i in range(len(listImg)):
        tmpImg = sky_trfm.resize(listImg[i], (64,64))
        io.imsave(listNameImg[i], tmpImg)
        
def enhancedAllImg(tabImg,filenameImg):
    for i in range(len(tabImg)):
        tabImg[i] = sk_util.img_as_ubyte(tabImg[i]) # Ne fait rien
        trfImg.modifImageRotate(tabImg[i],filenameImg[i])
        trfImg.modifImageCrop(tabImg[i],filenameImg[i]) #Toujours en dernier

