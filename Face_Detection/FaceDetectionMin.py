import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0) #0 for webcam, "Face_Detection\Face_1.mp4" RELATIVE PATH
previousTime = 0


mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75) #detection cofidence (we increase it to avoid false detection), the video here is not the best one XD

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            #print(id, detection)
            print(detection.score)
            print(detection.location_data.relative_bounding_box)
            boundingBoxClass = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            boundingBox = int(boundingBoxClass.xmin * w), int(boundingBoxClass.ymin * h), \
                        int(boundingBoxClass.width * w), int(boundingBoxClass.height * h)
            
            cv2.rectangle(img, boundingBox, (255, 0, 0), 2)
            currentTime = time.time()
            fps = int(1/(currentTime - previousTime))
            previousTime = currentTime
            #With round(___ , 2) we limit to the second decimal number
            #now we are printing the score in percentage. We are removing 20 for a better view
            cv2.putText(img, f"{round(detection.score[0] * 100, 2)} %",
                         (boundingBox[0], boundingBox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)




    
    cv2.putText(img, f"FPS: {fps}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(5)
