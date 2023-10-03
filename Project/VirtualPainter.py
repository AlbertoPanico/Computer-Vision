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
drawColor = (255, 0, 255)
brushTickness = 15

cap = cv2.VideoCapture(0)
cap.set(3, 1500)
cap.set(4, 720)
xPrevious, yPrevious = 0, 0
imgCanvas = np.zeros((720, 1500, 3), np.uint8)

detector = htm.HandDetector(detectionConfidence=0.85)

while True:
    # Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks
    img = detector.findHands(img)
    landMarkList = detector.findPosition(img, draw=False)
    if len(landMarkList) != 0:
        # tip of index (id = 8) and middle (id = 12)
        xIndex, yIndex = landMarkList[8][1:]
        xMiddle, yMiddle = landMarkList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # Select mode (fingers up)
        if fingers[1] and fingers[2]:

            print("Selection mode")
            if yIndex < 125:
                # if true we are in the header
                if 0 < xIndex < 250:
                    # blue
                    drawColor = (255, 0, 0)
                    print(1)
                elif 250 < xIndex < 550:
                    # green
                    print(2)
                    drawColor = (0, 255, 0)
                elif 580 < xIndex < 900:
                    # red
                    print(3)
                    drawColor = (0, 0, 255)
                elif 900 < xIndex < 1144:
                    # eraser (black)
                    print(4)
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (xIndex, yIndex - 25), (xMiddle, yMiddle + 25), drawColor, cv2.FILLED)

        # Draw mode(index up)
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (xIndex, yIndex), 15, (255, 0, 255), cv2.FILLED)
            print("Drawing mode")
            if xPrevious and yPrevious == 0:
                xPrevious, yPrevious = xIndex, yIndex
            if drawColor == (0, 0, 0):
                #if we select the eraser, we increase the tickness
                brushTickness = 30
            cv2.line(img, (xPrevious, yPrevious), (xIndex, yIndex), drawColor, brushTickness)
            cv2.line(imgCanvas, (xPrevious, yPrevious), (xIndex, yIndex), drawColor, brushTickness)
            xPrevious, yPrevious = xIndex, yIndex

    # Setting the header image
    heightPic, widthPic, channelPic = overlayList[0].shape
    img[0:heightPic, 0:widthPic] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Image Canvas", imgCanvas)
    cv2.waitKey(1)
