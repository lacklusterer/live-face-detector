import os
import argparse
from helpers.file_helper import get_image, write_image

def main(image_path, model, debug):
    try:
        image, input_image_path = get_image(image_path, debug)
        if debug:
            print(f"[DEBUG] Image loaded: {input_image_path}")

        # Model selection
        if debug:
            print(f"[DEBUG] Model selected: {model}")

        if model == 'haarcascade':
            output_image = haarcascades_process(image)
        elif model == 'yunet':
            output_image = yunet_process(image)
        else:
            raise ValueError("Invalid model selected. Choose 'haarcascade' or 'yunet'.")

        output_image_path = write_image(output_image, debug)    
        if debug:
            print(f"[DEBUG] Processed image saved to {output_image_path}")
    
    except Exception as e:
        print(e)

def haarcascades_process(frame):
    from models.haarcascade.haarcascades import process
    return process(frame)

def yunet_process(frame):
    from models.yunet.yunet import process
    return process(frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing script")
    parser.add_argument('--image', type=str, help='Path to the input image.')
    parser.add_argument('--model', type=str, choices=['haarcascade', 'yunet'], help='Select the face detection model.', required=True)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print additional messages.')
    
    args = parser.parse_args()
    main(args.image, args.model, args.debug)

