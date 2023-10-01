import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, minDetectionConfidence = 0.75):
        self.minDetectionConfidence = minDetectionConfidence
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionConfidence)

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        #print(self.results)
        boundingBoxsList = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundingBoxClass = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                boundingBox = int(boundingBoxClass.xmin * w), int(boundingBoxClass.ymin * h), \
                        int(boundingBoxClass.width * w), int(boundingBoxClass.height * h)
                boundingBoxsList.append([id, boundingBox, detection.score])
                if draw:
                    img = self.fancyDraw(img, boundingBox)
                
                    cv2.putText(img, f"{round(detection.score[0] * 100, 2)} %",
                         (boundingBox[0], boundingBox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        
        return img, boundingBoxsList
    
    def fancyDraw(self, img, boundingBox, l=30):
        x, y, w, h = boundingBox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, boundingBox, (255, 0, 0), 2)
        #top left x, y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 0), 5)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 0), 5)
        #top right x1, y
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 0), 5)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 0), 5)
        #botton left x, y1
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 0), 5)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 0), 5)
        #botton right x1, y1
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 0), 5)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 0), 5)
        return img


def main():
    cap = cv2.VideoCapture(0) #0 for webcam, "Face_Detection\Face_1.mp4" RELATIVE PATH
    previousTime = 0
    detector = FaceDetector()

    
    while True:
        success, img = cap.read()
        img, boundingBoxsList = detector.findFaces(img)
   
        currentTime = time.time()
        fps = int(1/(currentTime - previousTime))
        previousTime = currentTime
        cv2.putText(img, f"FPS: {fps}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(5)



if __name__ == "__main__":
    main()