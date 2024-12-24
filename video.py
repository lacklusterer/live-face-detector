import cv2
import os
from helpers.file_helper import write_image
from helpers.video_display import display_frame
from models.haarcascade.haarcascades import haarcascades_process

def process_video_stream(debug=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Cannot open webcam.")

    if debug:
        print("[DEBUG] Webcam stream opened successfully.")
    
    # Set FPS for the capture to match the video writer
    cap.set(cv2.CAP_PROP_FPS, 20)

    # Create output folder if it doesn't exist
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Set video filename and codec
    video_filename = os.path.join(output_folder, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 files
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))  # 20.0 FPS, 640x480 resolution
    
    frame_counter = 0
    max_frame = 100

    while frame_counter <= max_frame:
        ret, frame = cap.read()
        if not ret:
            if debug:
                print("[DEBUG] Failed to capture frame. Exiting...")
            break
        
        processed_frame = haarcascades_process(frame, debug)

        # Write the processed frame to video file
        out.write(processed_frame)
        frame_counter += 1
        if debug:
            print(f"[DEBUG] Frame: {frame_counter}/{max_frame}")

    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video stream face detection script")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print additional messages.')

    args = parser.parse_args()
    process_video_stream(args.debug)

