import cv2
import sys
import os

def get_image(image_path=None):
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    
    if not image_path:
        print("No image path supplied, capturing with webcam ...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Error: Cannot open webcam.")
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("Error: Cannot capture image from webcam.")
        
        input_image_path = os.path.join("tmp", "captured_image.jpg")
        cv2.imwrite(input_image_path, frame)
        print(f"Captured image saved to: {input_image_path}")
    else:
        input_image_path = image_path
        print(f"Image path supplied: {input_image_path}")
    
    image = cv2.imread(input_image_path)
    if image is None:
        raise Exception("Error: Cannot load image.")
    
    return image, input_image_path
