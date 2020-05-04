import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import os

#basic set-up of camera and classifier
camera=PiCamera()
camera.resolution = (640,480)
camera.framerate=32

sampleCapture= PiRGBArray(camera, size=(640,480))

classifier =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classifier.load('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml') #the address of classifier
# create the dir for data storage, also check duplication
create=True
while(create):
    
    user = input("What's your name?")
    userdir= "./datasets/" +user

    if os.path.exists(userdir):
        print(" Sorry, the name has been used! Please input another name.")
        
    else:
        os.makedirs(userdir)
        print("Your directory created!")
        create=False
    
print("Welcome to data gathering part")


count=0
# capture the picture and save it with classifier, in this case, we gather 100 )
for frame in camera.capture_continuous(sampleCapture, format="bgr", use_video_port=True):
    
    
    frame=frame.array
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces=classifier.detectMultiScale(gray,scaleFactor=1.5,minNeighbors =5,)

    for (x,y,w,h) in faces:
        roiGray = gray[y:y+h ,x:x+w]
        picName=userdir + "/" +user + str(count) + ".jpg"
        cv2.imwrite(picName, roiGray)
        cv2.imshow("face",roiGray)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        
        count +=1
    
    cv2.imshow('frame',frame)
    button=cv2.waitKey(1)
    
    sampleCapture.truncate(0)
    
    if (button ==27) or (count >49):
        print(" Data gathering successfully!!")
        break
cv2.destroyAllWindows()
