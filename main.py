import os
from capture import get_image
from process_face import process_image

def main():
    try:
        image, input_image_path = get_image()
        print(f"Image loaded: {input_image_path}")
        _, output_image_path = process_image(image)
        print(f"Processed image saved to {output_image_path}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
