import mediapipe as mp
import time 
import cv2

cap = cv2.VideoCapture(0)
previousTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for landmark in faceLandmarks.landmark:
                #print(landmark)
                h, w, h = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                print(x, y)
                




    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(5)