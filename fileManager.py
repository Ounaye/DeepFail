# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:39:54 2021

@author: Ounaye
"""

import glob as glb
from skimage import io
import skimage.transform as sky_trfm

import transformImage as trfImg
import skimage.util as sk_util

# Si on enlève le filtrage .jpeg 
# Il a besoin de format de lecture en arg

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



def rescaleImg(listImg,listNameImg):
#Flemme de gérer le warning pour l'instant
    for i in range(len(listImg)):
        tmpImg = sky_trfm.resize(listImg[i], (256,256))
        io.imsave(listNameImg[i], tmpImg)
        
def enhancedAllImg(tabImg,filenameImg):
    for i in range(len(tabImg)):
        tabImg[i] = sk_util.img_as_ubyte(tabImg[i]) #useless ?
        trfImg.modifImageRotate(tabImg[i],filenameImg[i])
        trfImg.modifImageCrop(tabImg[i],filenameImg[i]) #Toujours en dernier
