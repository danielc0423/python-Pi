from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import pickle

#pi camera setting                                              
camera =PiCamera()
camera.resolution=(640,480)
camera.framerate=40
realtimeCapture =PiRGBArray(camera,size=(640,480))


with open('labels','rb') as f:
    dict=pickle.load(f)
    f.close()
#classifier and trianer

classifier =cv2.CascadeClassifier("harrcascade_frontalface_default.xml")
classifier.load('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

font=cv2.FONT_HERSHEY_SIMPLEX

for frame in camera.capture_continuous(realtimeCapture, format=("bgr"),use_video_port=True):
    frame=frame.array
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=classifier.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    
    for (x, y, w,h) in faces:
        
        roiwG =gray[y:y+h,x:x+w]
        
        user,difference=recognizer.predict(roiwG)
        
        if difference<=70:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            print("welcome")
        else:   
            print(" YOU DON'T HAVE ACCESS  ")
                

    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)

    realtimeCapture.truncate(0)
    if key == 27:
        break
    
cv2.destroyAllWindows()
