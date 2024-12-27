import cv2

def get_bounding_boxes(image, debug=False):
    face_cascade = cv2.CascadeClassifier('models/haarcascade/haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if debug:
        print(f"Bounding Boxes: {bounding_boxes}")
    return bounding_boxes

