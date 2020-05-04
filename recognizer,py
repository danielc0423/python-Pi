import cv2
import os
import numpy as np
from PIL import Image
import pickle

recognizer =cv2.face.LBPHFaceRecognizer_create()
classifier =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classifier.load('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

file_dir = os.path.dirname(os.path.abspath(__file__))
pic_dir =os.path.join(file_dir,"datasets")
currentId =1
labelIds={}
yLabels =[]
xTrain =[]

for root,dirs,files in os.walk(pic_dir):
    print(root,dirs,files)
    for file in files:
        print(file)
        
        if file.endswith("png")or file.endswith("jpg"):
            
            path =os.path.join(root ,file)
            label = os.path.basename(root)
            print(label)
            
            if not label in labelIds:
                labelIds[label] =currentId
                print(labelIds)
                currentId += 1
                
                
            id_ =labelIds[label]
            pilImage= Image.open(path).convert("L")
            imageArray = np.array(pilImage , "uint8")
            faces =classifier.detectMultiScale(imageArray,scaleFactor=1.5, minNeighbors =5)
            
            for(x,y,w,h) in faces:
                roi = imageArray[y:y+h,x:x+w]
                xTrain.append(roi)
                yLabels.append(id_)
                
with open("labels","wb") as f:
    pickle.dump(labelIds,f)
    f.close
    
recognizer.train(xTrain,np.array(yLabels))
recognizer.save("trainer.yml")
print(labelIds)
