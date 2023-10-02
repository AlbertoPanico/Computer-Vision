import cv2
import mediapipe as mp
import time
import os
import Hand_Tracking.HandTrackingModule as htm

widthCam, heightCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)

folderPath = "img"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imagePath in myList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    #print(f'{folderPath}/{imagePath}')
    overlayList.append(image)
#print(len(overlayList))
previousTime = 0

#create a detector
detector = htm.HandDetector(detectionConfidence=0.75)

#these are the ids of the heighest part of the fingers
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landMarkList = detector.findPosition(img, draw=False)
    #print(landMarkList)
    if len(landMarkList) != 0:
        fingers = []

        #thumb detection (if point 4 is on the right of the point 3, it will be assumed as close)
        id = 0
        if landMarkList[tipIds[id]][1] > landMarkList[tipIds[id] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #detect if the thumb is open or not is not like other fingers, so we start from the second one
        for id in range(1, 5):
            #on mediapipe's website there are a picture with the number of every part of the hand, so you can understand
            if landMarkList[tipIds[id]][2] < landMarkList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        heightPic, widthPic, channelPic = overlayList[totalFingers].shape
        #limit of the  width and height
        img[0:heightPic, 0:widthPic] = overlayList[totalFingers]

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)