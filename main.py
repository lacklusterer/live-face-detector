import os
import argparse
from helpers.file_helper import get_image, write_image
from models.haarcascade.haarcascades import haarcascades_process

def main(image_path, debug):
    try:
        image, input_image_path = get_image(image_path, debug)
        if debug:
            print(f"[DEBUG] Image loaded: {input_image_path}")

        output_image = haarcascades_process(image, debug)
        output_image_path = write_image(output_image, debug)    
        if debug:
            print(f"[DEBUG] Processed image saved to {output_image_path}")
    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing script")
    parser.add_argument('--image', type=str, help='Path to the input image.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print additional messages.')
    
    args = parser.parse_args()
    main(args.image, args.debug)

