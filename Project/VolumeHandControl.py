import cv2
import math
import time
import numpy as np
from Hand_Tracking.HandTrackingModule import HandDetector

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

widthCam, heightCam = 640, 480

#pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)
minVolume = volumeRange[0]
maxVolume = volumeRange[1]

#check if webcam is working
cap = cv2.VideoCapture(0)
previousTime = 0
#hand tracking module
detector = HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landMarkList = detector.findPosition(img, draw = False)
    if len(landMarkList) != 0:    
        #4 and 8 are the point we want to know to detect the gesture 
        #print(landMarkList[4], landMarkList[8])

        #for thumb
        x_thumb, y_thumb = landMarkList[4][1], landMarkList[4][2]
        #for index
        x_index, y_index = landMarkList[8][1], landMarkList[8][2]

        # now create a line between the two fingers
        cv2.line(img, (x_thumb, y_thumb), (x_index, y_index), (255, 0, 255), 3)

        #get the center of this line
        center_x, center_y = (x_thumb + x_index)//2, (y_thumb + y_index)//2

        cv2.circle(img, (x_thumb, y_thumb), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x_index, y_index), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (center_x, center_y), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x_index - x_thumb, y_index - y_thumb)
        print(length)

        #hand range -144 : .40
        #volume range  -65 : 0
        vol = np.interp(length, [15, 190], [minVolume, maxVolume])
        print(vol)
        # now change the volume
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (center_x, center_y), 15, (0, 255, 0), cv2.FILLED)




    

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
    cv2.imshow("Img", img)
    cv2.waitKey(1)