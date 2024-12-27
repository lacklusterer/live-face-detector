import cv2

def get_annote(image, debug=False):
    height, width = image.shape[:2]
    face_detector = cv2.FaceDetectorYN_create('models/yunet/face_detection_yunet_2023mar.onnx', "", (width, height))
    _, faces = face_detector.detect(image)
    
    annote = []
    if faces is not None:
        annote = faces
        if debug:
            print(annote)

    return annote
