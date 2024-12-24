import os
from file_helper import get_image, write_image
from process_face import process_image

def main():
    try:
        image, input_image_path = get_image()
        print(f"Image loaded: {input_image_path}")

        output_image = process_image(image)
        output_image_path = write_image(output_image)    
        print(f"Processed image saved to {output_image_path}")
    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
