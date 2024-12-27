import cv2

def get_annote(image, debug=False):
    height, width = image.shape[:2]
    face_detector = cv2.FaceDetectorYN_create('models/yunet/face_detection_yunet_2023mar.onnx', "", (width, height))
    _, faces = face_detector.detect(image)
    if faces is None:
        if debug:
            print("No faces detected.")
        return []
    if debug:
        print(faces)
    return faces

def get_bounding_boxes(image, debug=False):
    faces = get_annote(image, debug)
    bounding_boxes = [face[:4] for face in faces]
    if debug:
        print(f"Bounding Boxes: {bounding_boxes}")
    return bounding_boxes
