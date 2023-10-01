import mediapipe as mp
import time 
import cv2

class FaceMeshDetector():
    def __init__(self, staticImageMode = False, maxFaces = 2, minDetectionConfidence = 0.5, minTrackingConfidence = 0.5):
        self.staticImageMode = staticImageMode
        self.maxFaces = maxFaces
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence


        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

    def findFaceMesh(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            
            for faceLandmarks in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS,
                                        self.drawSpec, self.drawSpec)
                    face = []
                    for id, landmark in enumerate(faceLandmarks.landmark):
                        h, w, c = img.shape
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        #print(id, x, y)
                        face.append([x, y])
                    faces.append(face)
        return img, faces
              
                

def main():
    cap = cv2.VideoCapture(0)
    previousTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(5)


if __name__ == "__main__":
    main()