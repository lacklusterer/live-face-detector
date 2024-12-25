import cv2
import os

def process(frame, debug=False):
    image = cv2.flip(frame, 1)

    face_cascade = cv2.CascadeClassifier('models/haarcascade/haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if debug:
        print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

