import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectionConfidence = 0.5, trackConfidence = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]




    def findHands(self, img, draw = True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #check if we have multiple hands
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    


    def findPosition(self, img, handNumber = 0, draw = True):
          
        self.landMarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                center_x, center_y = int(lm.x * w), int(lm.y * h)
                #print(id, center_x, center_y)
                self.landMarkList.append([id, center_x, center_y])
                if draw: 
                    cv2.circle(img, (center_x, center_y), 7, (255, 0, 0), cv2.FILLED)
        
        
        return self.landMarkList

    def fingersUp(self):
        fingers = []

        # thumb detection (if point 4 is on the right of the point 3, it will be assumed as close)
        id = 0
        if self.landMarkList[self.tipIds[id]][1] > self.landMarkList[self.tipIds[id] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # detect if the thumb is open or not is not like other fingers, so we start from the second one
        for id in range(1, 5):
            # on mediapipe's website there are a picture with the number of every part of the hand, so you can understand
            if self.landMarkList[self.tipIds[id]][2] < self.landMarkList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers



                

            
 

def main():
    
    previusTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4]) #4 is the end of the thumb (pollice)

        
        
        currentTime = time.time()
        fps = 1/(currentTime - previusTime)
        previusTime = currentTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()