import cv2
import json

def process(frame, debug=False):
    image = cv2.flip(frame, 1)
    height, width = image.shape[:2]

    face_detector = cv2.FaceDetectorYN_create('models/yunet/face_detection_yunet_2023mar.onnx', "", (width, height))
    _, faces = face_detector.detect(image)
    
    if faces is not None:
        face_data = []
        for face in faces:
            bounding_box = face[:4].astype(int).tolist()
            confidence = face[14]
            landmarks = {
                "left_eye": face[4:6].tolist(),
                "right_eye": face[6:8].tolist(),
                "nose": face[8:10].tolist(),
                "left_mouth": face[10:12].tolist(),
                "right_mouth": face[12:14].tolist()
            }

            face_data.append({
                "bounding_box": {
                    "x": bounding_box[0],
                    "y": bounding_box[1],
                    "width": bounding_box[2],
                    "height": bounding_box[3]
                },
                "confidence": confidence,
                "landmarks": landmarks
            })

            cv2.rectangle(
                image, 
                (bounding_box[0], bounding_box[1]), 
                (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), 
                (255, 0, 0), 
                2
            )

            for landmark in landmarks.values():
                cv2.circle(image, (int(landmark[0]), int(landmark[1])), 3, (0, 255, 0), -1)

            cv2.putText(
                image, 
                f"{confidence:.2f}", 
                (bounding_box[0], bounding_box[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 255), 
                1
            )

        if debug:
            for item in face_data:
                item['confidence'] = float(item['confidence'])
            print(f"[DEBUG] Face detected: {json.dumps(face_data, indent=4)}")

    return image

