import cv2
import numpy as np
import time
import os
import Hand_Tracking.HandTrackingModule as htm

folderPath = "header"
myList = os.listdir(folderPath)
overlayList = []
for imagePath in myList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1500)
cap.set(4, 720)

detector = htm.HandDetector(detectionConfidence=0.85)

while True:
    #Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #Find Hand Landmarks
    img = detector.findHands(img)
    landMarkList = detector.findPosition(img, draw=False)
    if len(landMarkList) != 0:
        # tip of index (id = 8) and middle (id = 12)
        xIndex, yIndex = landMarkList[8][1:]
        xMiddle, yMiddle = landMarkList[12][1:]

        #Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)



        #Select mode (fingers up)
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (xIndex, yIndex - 25), (xMiddle, yMiddle + 25), (255, 0, 255), cv2.FILLED)
            print("Selection mode")
            if yIndex < 125:
                #if true we are in the header
                if 0 < xIndex < 250:
                    print(1)
                elif 250 < xIndex < 550:
                    print(2)
                elif 580 < xIndex < 900:
                    print(3)
                elif 900 < xIndex < 1144:
                    print(4)



        #Draw mode(index up)
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (xIndex, yIndex), 15, (255, 0, 255), cv2.FILLED)
            print("Drawing mode")


    #Setting the header image
    heightPic, widthPic, channelPic = overlayList[0].shape
    img[0:heightPic, 0:widthPic] = header
    cv2.imshow("Image", img)
    cv2.waitKey(1)