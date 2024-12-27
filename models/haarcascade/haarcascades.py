import cv2
from models.haarcascade.haarcascade_annote import get_bounding_boxes

def process(frame, debug=False):
    image = cv2.flip(frame, 1)

    faces = get_bounding_boxes(image, debug)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

