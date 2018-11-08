# -*- coding: utf-8 -*-
import cv2
import numpy as np

oldArea=1
Z_mean = np.zeros(10)

def GetMoments(skinMask):
    C_new=None
    bw, contours,contourHierarchy= cv2.findContours(skinMask,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    Area = 0
    """pesquisar pela maior area"""
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if Area < area:
            Area = area
            C_new = cnt
    return C_new

def Moments(cnt):
    """retirar perimetro area"""
    P = np.sum(cv2.boundingRect(cnt))
    A = cv2.contourArea(cnt)
    (x, y), r = cv2.minEnclosingCircle(cnt)
    r = int(r)
    return A,P,r

def zoomInOut(skinMask):
    global oldArea
    global Z_mean
    mainSkin=GetMoments(skinMask)
    P, A, r = Moments(mainSkin)
    Z_mean = np.roll(Z_mean,1)
    #print (Z_mean)
    #print (oldArea-A)
    if (oldArea-A)<-5:
        Z_mean[0] = -1
    elif (oldArea-A)>5:
        Z_mean[0] = 1
    else:
        Z_mean[0] = 0

    if np.mean(Z_mean)<0:
        zoom = "IN"
    elif np.mean(Z_mean)>0:
        zoom = "OUT"
    else:
        zoom = ""
    oldArea = A
    return r,zoom
    






