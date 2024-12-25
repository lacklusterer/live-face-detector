import cv2

def process(image):
    # Initialize the face detector
    face_detector = cv2.FaceDetectorYN_create('models/yunet/face_detection_yunet_2023mar.onnx', "", (640, 480))
    
    # Detect faces
    _, faces = face_detector.detect(image)  # Pass the BGR image
    
    # Check if any faces are detected
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)  # Extract bounding box coordinates
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return image

