# -*- coding: utf-8 -*-
import cv2

oldfeuturesToTrack=None
newfeuturesToTrack=None

def calcFeatures(skinImg,lastSkinImg):
    """ATENCAO codigo muito sensivel minima mudança deixa de funcionar"""
    global color
    global oldfeuturesToTrack
    global newfeuturesToTrack
    good_old=None
    good_new=None
    st=None
    """feature_params e lk_params parametros copiados da net- ver https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html"""
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    """goodFeuturesTrack funciona com imagens em tons de cinzento"""
    lastSkinGray = cv2.cvtColor(lastSkinImg, cv2.COLOR_BGR2GRAY)
    skinGray = cv2.cvtColor(skinImg, cv2.COLOR_BGR2GRAY)
    
    """filtro para aumentar os contornos como o goodfeuturesTrack baseia nos contornos para obter bons pontos 
    melhor o contraste melhor é os pontos"""
    skinGray = cv2.bilateralFilter(skinGray,9,1,100)
    lastSkinGray= cv2.bilateralFilter(lastSkinGray,9,1,100)
    
    """o calculo de novos good feutures é so feito ao inicio ou quando os antigos sao perdidos ou sao inferiores a 5"""
    if oldfeuturesToTrack is None or len(oldfeuturesToTrack)<=5 or newfeuturesToTrack is None:
        oldfeuturesToTrack =cv2.goodFeaturesToTrack(lastSkinGray, mask = None, **feature_params)
        
    """calcOpticalFlowPyrLK() pesquisa os pontos do frame anterior (oldFeutures) no novo frame e retorna os pontos que 
    foram encontrados (newFeutures) os perdidos o st fica a 0 e os encontrados fica a 1"""
    if oldfeuturesToTrack is not None:
        newfeuturesToTrack, st, err = cv2.calcOpticalFlowPyrLK(lastSkinGray,skinGray, oldfeuturesToTrack, None, **lk_params)
        # Select good points
    if newfeuturesToTrack is not None and  oldfeuturesToTrack is not None :
        """atualizar os pontos existentes os que st for 0 sao eliminados"""
        good_new = newfeuturesToTrack[st==1]
        good_old = oldfeuturesToTrack[st==1]
            
        oldfeuturesToTrack = good_new.reshape(-1,1,2)
    return good_old,good_new


def calcDirection(good_old,good_new):
    """calcula  a direçao do movimento de acordo com a distancia entre os dois pontos
    se a distancia em x for negativa quer dizer que move se para a direita se for positiva esquerda e o mesmo para y
    
    faz uma media mas NAO ESTA A FUNCIONAR a 100%"""
    counterX=0
    counterY=0
#    print(newPoints)
    for i,(startP,endP) in enumerate(zip(good_old,good_new)):
        a,b = startP.ravel()
        c,d = endP.ravel()
        
        distX=c-a
        distY=d-b
        
        counterX+=distX
        counterY+=distY
        
    counterX=counterX/len(good_old)
    counterY=counterY/len(good_old)
    
    horizontal = ""
    if counterX<-10:
        horizontal = "Direita"
    elif counterX>10:
        horizontal = "Esquerda"

    vertical = ""
    if counterY<-10:
        vertical = "Cima"
    elif counterY>10:
        vertical = "Baixo"
    return horizontal, vertical
