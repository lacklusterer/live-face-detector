import cv2
import os

def haarcascades_process(image):
    if not os.path.exists("output"):
        os.makedirs("output")
    
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    
    output_image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(faces)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image

