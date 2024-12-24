import os
from file_helper import get_image, write_image
from haarcascades import haarcascades_process

def main():
    try:
        image, input_image_path = get_image()
        print(f"Image loaded: {input_image_path}")

        output_image = haarcascades_process(image)
        output_image_path = write_image(output_image)    
        print(f"Processed image saved to {output_image_path}")
    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
