# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 11:01:32 2019

@author: abhishek
"""

import cv2
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')
cap=cv2.VideoCapture(0)
while(True):
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi=frame[y:y+h,x:x+w]
        roi_gray=gray[y:y+h,x:x+w]
        
        eye=eye_cascade.detectMultiScale(roi_gray,1.3,5)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smile=smile_cascade.detectMultiScale(roi_gray,1.3,23)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    cv2.imshow("Video",frame)
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
