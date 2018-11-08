# -*- coding: utf-8 -*-
import numpy as np
import cv2

"""estes valores sao dados em hardCode pode ser necessario m
de acordo com o ton de pele da pessoa estes valores sao bastante 
sensiveis de pessoa para pessoa"""
lower = np.array([0, 35, 80], dtype = "uint8")
upper = np.array([40, 255, 255], dtype = "uint8")

Rnorm=None
Gnorm=None

def calcRGnorm(square):
    global Rnorm
    global Gnorm
    """Obter a pele"""
#    converter para RG norm
    showImgFromMemory(square)
    
    b,g,r = cv2.split(square)
#    print(r)

#    normalizacao     
    r=r.astype(int)
    g=g.astype(int)
    b=b.astype(int)
    
    RnormArr=r/(r+g+b)
    GnormArr=g/(r+g+b)
#    print(RnormArr)
    
    RnormArr=np.around(RnormArr, decimals=3)
    GnormArr=np.around(GnormArr, decimals=3)
    
    Rnorm=np.mean(RnormArr.ravel())
    Gnorm=np.mean(GnormArr.ravel())
    
    print(Rnorm)
    print(Gnorm)

#    para mostrar imagens em memoria
def showImgFromMemory(image, name = ""):
    cv2.imshow("Image "+name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def skinTrackRG(img):
#    debug  apagar depois
    global Rnorm
    global Gnorm
    global skinG
    global skinR
    """Obter a pele"""
#    converter para RG norm
    b,g,r = cv2.split(img)
    
    r=r.astype(float)
    g=g.astype(float)
    b=b.astype(float)
#    normalizacao RGnormalizado
    RnormArr=r/(r+g+b)
    GnormArr=g/(r+g+b)
#    lower e upper é o intervalo do ton da pele mais facil de encontrar
#    dia
    """valores da pele"""
    skinG = cv2.inRange(GnormArr,Gnorm-0.05,Gnorm+0.05)
    skinR = cv2.inRange(RnormArr,Rnorm-0.05,Rnorm+0.05)
    
#    cv2.imshow("G",skinG)
#    cv2.imshow("R",skinR)
    
    skin=cv2.bitwise_and(skinG,skinR)
    
#    operaçoes morfologicas
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opening = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    dilate=cv2.dilate(opening,kernel)
    
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    
#    cv2.imshow("skinTrackRG",closing)
    return closing 

# Funcao nao e usada, serve para seguir a pele usando HSV
def skinTrackHSV(img):
    """Seguir a pele da mao"""
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#    lower e upper e o intervalo do ton da pele mais facil de encontrar
    skinMask = cv2.inRange(converted, lower, upper)
#    operacoes morfologicas    
    blur = cv2.bilateralFilter(skinMask,9,1,100)
#    operaçoes morfologicas
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    dilate=cv2.dilate(opening,kernel)
    
    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("skinTrackHSV",closing)
    return closing
