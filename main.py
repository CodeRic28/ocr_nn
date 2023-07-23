import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

###########################################################
path = 'myData'
testRatio = 0.2
##########################################################

count = 0
images = []
classNo = []

myList = os.listdir(path)
print("Total number of classes detected: ",myList)
noOfClasses = len(myList)
print("Importing Classes...")
for x in range(0, noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(x)
    print(count,end=" ")
    count+=1
print(" ")

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)

X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio)