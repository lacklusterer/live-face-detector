import os
from capture import get_image
from process_face import process_image

def main():
    output_folder = "output"
    try:
        image, input_image_path = get_image(output_folder)
        print(f"Image loaded: {input_image_path}")
        output_image_path = process_image(image, output_folder)
        print(f"Processed image saved to {output_image_path}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
