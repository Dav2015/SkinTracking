# -*- coding: utf-8 -*-
import numpy as np
import cv2

# Create some random colors
color = np.random.randint(0,255,(100,3))

def drawTrackPoints(good_old,good_new,localFrame):
    # draw the tracks
    """mostrar os pontos"""
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        localFrame= cv2.line(localFrame, (a,b),(c,d), color[i].tolist(), 2)
        localFrame= cv2.circle(localFrame,(a,b),5,color[i].tolist(),-1)
    cv2.imshow("track",localFrame)
    
def play(points,localFrame,horizontal,vertical,radius,zoom):  
    """ponto centro  para o circulo"""
    point=np.mean(points,axis=0)
    x=point[0]
    y=point[1]
    whiteImage=np.ones((localFrame.shape), np.uint8)*255
    cv2.circle(whiteImage,(x,y), radius, (0,0,255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(whiteImage,"Horizontal:"+str(horizontal),(25,25),\
                font,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(whiteImage,"Vertical:"+str(vertical),(200,25),\
                font,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(whiteImage,"Zoom:"+str(zoom),(375,25),\
                font,0.5,(0,0,0),1,cv2.LINE_AA)    
    cv2.imshow("circle",whiteImage)
    
    
