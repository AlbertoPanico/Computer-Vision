import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode = False, model_complexity = 1, smooth_landmarks = 1,
                 enable_segmentation = False, smooth_segmentation = False,
                 min_detection_confidence = 0.5,  min_tracking_confidence=0.5):
       
       self.mode = mode
       self.model_complexity = model_complexity
       self.smooth_landmarks = smooth_landmarks
       self.enable_segmentation = enable_segmentation
       self.smooth_segmentation = smooth_segmentation
       self.min_detection_confidence = min_detection_confidence
       self.min_tracking_confidence = min_tracking_confidence

       '''
       static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5
       '''

       self.mpDraw = mp.solutions.drawing_utils
       self.mpPose = mp.solutions.pose
       self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks,
                                    self.enable_segmentation, self.smooth_segmentation,
                                    self.min_detection_confidence, self.min_tracking_confidence)
       

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
        

    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                center_x, center_y = int(lm.x*w), int(lm.y*h)
                lmList.append([id, center_x, center_y])
                if draw: 
                    cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), cv2.FILLED)
        
        return lmList

       
        
    
    
def main():
    cap = cv2.VideoCapture('PoseVideos_1.mp4')
    previousTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw = False)
        print(lmList)

    
        currentTime = time.time()
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(5)

 
if __name__ == "__main__":
    main()