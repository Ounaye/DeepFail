# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:45:17 2021

@author: malig
"""
from PIL import Image
from PIL import ImageEnhance

def modifImageRotate(image):
    im = Image.open(image)
    out = im.rotate(45)
    out.show()

def modifImageCrop(image):
    im = Image.open(image)
    width, height = im.size
    left = width/8
    top = height/8
    right = 7*width/8
    bottom = 7*height/8
    out = im.crop((left, top, right, bottom))
    out.show()
    
def modifImageConstrast(image):
    im = Image.open(image)
    enh = ImageEnhance.Contrast(im)
    enh.enhance(1.3).show("30% more contrast")
    enh.show()
    

modifImageRotate("C:/Users/malig/Documents/Apprentissage automatique/DeepFail-main/DeepFail-main/perroquet.png")

modifImageCrop("C:/Users/malig/Documents/Apprentissage automatique/DeepFail-main/DeepFail-main/perroquet.png")

modifImageConstrast("C:/Users/malig/Documents/Apprentissage automatique/DeepFail-main/DeepFail-main/perroquet.png"