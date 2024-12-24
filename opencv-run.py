import cv2
import sys
import os

output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

n = len(sys.argv)

if n == 1:
    print("No image path supplied, capturing with webcam ...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        exit()
    print("Captured image")
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot capture image from webcam.")
        cap.release()
        exit()

    cap.release()
    input_image = os.path.join(output_folder, "captured_image.jpg")
    cv2.imwrite(input_image, frame)
else:
    input_image = sys.argv[1]

image = cv2.imread(input_image)
if image is None:
    print("Error: Cannot load image.")
    exit()

# Resize image (optional) for faster processing
image = cv2.resize(image, (640, 480))

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image using OpenCV
image = cv2.imread(input_image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the output image
output_image_path = os.path.join(output_folder, "processed_image.jpg")
cv2.imwrite(output_image_path, image)
print(f"Processed image saved to {output_image_path}")
