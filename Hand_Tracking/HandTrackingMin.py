import cv2
import mediapipe as mp
import time

#video object
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previusTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #check if we have multiple hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #lm = landmark
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                #center_x, center_y
                cx, cy = int(lm.x * w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    #we are print thr botton of the hand in purple
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    currentTime = time.time()
    fps = 1/(currentTime - previusTime)
    previusTime = currentTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)