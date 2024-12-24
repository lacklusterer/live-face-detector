import cv2
import os

def process_image(image, output_folder="output"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_image = cv2.resize(image, (640, 480))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_image_path = os.path.join(output_folder, "processed_image.jpg")
    cv2.imwrite(output_image_path, output_image)
    
    return output_image, output_image_path
