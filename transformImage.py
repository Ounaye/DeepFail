# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:45:17 2021
@author: malig
"""
from PIL import Image
from PIL import ImageEnhance

from matplotlib import pyplot as plt
from skimage.transform import rotate
from skimage.util import crop
from skimage import io

#Toutes les Images sont sous format jpeg


def modifImageRotate(img,filename): # Rotation de -3° à 3° par img 
    filename = filename[:(len(filename)-5)] #On enlève le .jpeg
    for i in range(-3,3):
        out = rotate(img,3)
        out = crop(out, ((2, 2), (0, 0), (0,0)), copy=False)
        io.imsave(filename+"R"+str(i)+".jpeg",out)

def modifImageCrop(img,filename):
    filename = filename[:(len(filename)-5)] #On enlève le .jpeg

    out = crop(img, ((4, 0), (0, 0), (0,0)), copy=False)
    io.imsave(filename+"T1.jpeg",out)
    out = crop(img, ((0, 4), (0, 0), (0,0)), copy=False)
    io.imsave(filename+"T2.jpeg",out)
    out = crop(img, ((0, 0), (4, 0), (0,0)), copy=False)
    io.imsave(filename+"T3.jpeg",out)
    out = crop(img, ((0, 0), (0, 4), (0,0)), copy=False)
    io.imsave(filename+".jpeg",out) #Ecrase l'image
    
def modifImageConstrast(image): # Pas encore fait
    im = Image.open(image)
    enh = ImageEnhance.Contrast(im)
    #enh.enhance(1.3).show("30% more contrast")
    plt.imshow(enh)



