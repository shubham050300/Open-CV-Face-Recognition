import os
import cv2
import numpy as np
from PIL import Image       #python image library

recognizer=cv2.createLBPHFaceRecognizer();
path='dataSet'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #to get all the images from the dataSet directory

    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L'); #converting to grayscale
        faceNp=np.array(faceImg,'uint8')  #converting PIL form to numpy array as opencv requires numpy dataset only
                                # unsigned integer 8
        ID=int(os.path.split(imagePath)[-1].split('.')[1]) #to get the corresponding user no.
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces

IDs,faces=getImagesWithID(path)
recognizer.train(faces,IDs)
recognizer.save('recognizer/traniningData.yml')
cv2.destroyAllWindows()
