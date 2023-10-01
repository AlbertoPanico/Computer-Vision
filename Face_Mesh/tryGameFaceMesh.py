import cv2

import time
import FaceMeshModule as fmm

previusTime = 0
currentTime = 0
cap = cv2.VideoCapture(0)
detector = fmm.FaceMeshDetector()

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)

    if len(faces) != 0:
        print(len(faces))  # 4 is the end of the thumb (pollice)

    currentTime = time.time()
    fps = 1 / (currentTime - previusTime)
    previusTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
