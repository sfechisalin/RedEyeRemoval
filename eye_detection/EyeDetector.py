import cv2

class EyeDetector:
    def __init__(self ):
        self.__eye_cascade = cv2.CascadeClassifier('./eye_detection/haarcascade_eye.xml')


    def detect(self, image, face_coordinates):
        eye_coordinates = []
        for (x1, y1, x2, y2) in face_coordinates:
            w = x2 - x1
            h = y2 - y1
            roi_gray = image[y1: y1 + h, x1: x1 + w]
            eyes = self.__eye_cascade.detectMultiScale(roi_gray, 1.1, 6)
            face_c =  (x1, y1, w, h)
            eyes_c = []
            for (ex, ey, ew, eh) in eyes:
                eyes_c.append((ex, ey, ew, eh))
                cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_coordinates.append((face_c, eyes_c))
        cv2.imshow('image_with_eyes_detected', image)
        cv2.waitKey(0)
        return eye_coordinates

