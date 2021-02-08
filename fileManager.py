# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:39:54 2021

@author: Ounaye
"""

import glob as glb
from skimage import io
import skimage.transform as sky_trfm

# Si on enlève le filtrage .jpeg 
# Il a besoin de format de lecture en arg

def makeTabOfImg():
    list_filesNM = glb.glob("Data/Ailleurs/**/*.jpeg", recursive=True)
    image_listNM = []
    for filename in list_filesNM:
        image_listNM.append(io.imread(filename))
    
    list_filesM = glb.glob("Data/Mer/**/*.jpeg", recursive=True)
    image_listM = []
    for filename in list_filesM:
        image_listM.append(io.imread(filename))
    return (image_listNM,image_listM)



def rescaleImg(listImg,listNameImg):
#Flemme de gérer le warning pour l'instant
    for i in range(len(listImg)):
        tmpImg = sky_trfm.resize(listImg[i], (64,64))
        io.imsave(listNameImg[i], tmpImg)