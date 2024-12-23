
import cv2
import os
from retinaface import RetinaFace

# Input source: 0 for webcam or provide a video file path
input_source = 0  # Change to 'video.mp4' for a video file

# Create output folder if it doesn't exist
output_folder = "output_video"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize video capture
cap = cv2.VideoCapture(input_source)

fps = 24  # You can change this to match the video file FPS
cap.set(cv2.CAP_PROP_FPS, fps)

if not cap.isOpened():
    print("Error: Cannot open video source.")
    exit()

# Get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
output_filename = os.path.join(output_folder, "output_video.mp4")
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

frame_count = 0
frame_skip = 5  # Process every 5th frame to speed up

# Loop through video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error.")
        break

    # Skip frames to reduce load
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Resize frame to speed up processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert to RGB and detect faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("Retinafacing...")
    faces = RetinaFace.detect_faces(rgb_frame)

    if isinstance(faces, dict):  # If faces are detected
        for key, face_data in faces.items():
            facial_area = face_data['facial_area']
            landmarks = face_data['landmarks']

            # Draw bounding box around the face
            cv2.rectangle(frame, (facial_area[0], facial_area[1]), 
                          (facial_area[2], facial_area[3]), (0, 255, 0), 2)

            # Draw landmarks
            for point in landmarks.values():
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    # Write the frame to the video output file
    print("Writing to ouput file...")
    out.write(frame)

    frame_count += 1
    print(f"Processed frame: {frame_count}")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

