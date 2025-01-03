import cv2
import sys
import os
from datetime import datetime

def get_image(image_path=None, debug=False):
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    
    if not image_path:
        if debug:
            print("[DEBUG] No image path supplied, capturing with webcam")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Error: Cannot open webcam.")
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("Error: Cannot capture image from webcam.")
        
        input_image_path = os.path.join("tmp", "captured_image.jpg")
        cv2.imwrite(input_image_path, frame)
        if debug:
            print(f"[DEBUG] Captured image saved to: {input_image_path}")
    else:
        input_image_path = image_path
        if debug:
            print(f"[DEBUG] Image path supplied: {input_image_path}")
    
    image = cv2.imread(input_image_path)
    if image is None:
        raise Exception("Error: Cannot load image.")
    
    return image, input_image_path

def write_image(image, debug=False):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]
    output_image_path = os.path.join("output", f"{timestamp}.jpg")
    cv2.imwrite(output_image_path, image)
    
    return output_image_path

