import cv2
import numpy as np

import drawImages.showGraphics as draw
import skinMask.skinTrack as skin
import goodFeuturesTrack.track as track
import goodFeuturesTrack.caracteristics as caracte

def sequence():
    vid = cv2.VideoCapture(0)
    ret,lastSkinImage=vid.read()     
    while(True):
        # Capture frame-by-frame
        ret, frame = vid.read()
        """so corre e o frame estiver todo construido"""
        if ret == True:
            skinMask=skin.skinTrackRG(frame)
            
            """aplicar  a mascara a imagem  """     
            skinImage=putMaskToBGRImage(frame,skinMask) 
            good_old,good_new=track.calcFeatures(skinImage.copy(),lastSkinImage.copy())
            
            if good_new is not None and good_old is not None \
               and (len(good_new )and len(good_old))>1 and not np.all(skinImage==0):
                """calcular a direcao"""
                horizontal, vertical = track.calcDirection(good_old,good_new)
                radius,zoom=caracte.zoomInOut(skinMask)
                """circulo segue a posiçao da mao"""
                draw.drawTrackPoints(good_old,good_new,frame.copy())
                draw.play(good_new,frame,horizontal,vertical,radius,zoom)
                
            lastSkinImage=skinImage
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    vid.release()
    cv2.destroyAllWindows()
    
def setup():
    vid = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret,frame = vid.read()
        
        centerX=int(frame.shape[0]/2)
        centerY=int(frame.shape[1]/2)
        if ret==True:
            
            cv2.rectangle(frame,(centerX,centerY-150),
                          (centerX+100,centerY-100),(255,255,255),3)
            
            cv2.putText(frame,'Coloque pele no quadrado',
                        (100,50), font, 1,(255,255,255),2)
            
            cv2.imshow("setup",frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            crop_img = frame[centerY-150:centerY-100,
                             centerX:centerX+100]
            
            skin.calcRGnorm(crop_img.copy())
            #print(crop_img.shape)
            cv2.imshow("crop",crop_img)
            break
    # When everything done, release the capture
    vid.release()
    cv2.destroyAllWindows()
            
def putMaskToBGRImage(img, mask):
    """aplicar mascara binaria a uma imagem BGR"""
    justHand = cv2.bitwise_and(img,img,mask=mask)
    #cv2.imshow("hand",justHand)
    return justHand
    
setup()
sequence()
    
