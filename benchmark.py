import cv2
import os
import argparse
from helpers.benchmarker import *

def main():
    parser = argparse.ArgumentParser(description='Face Detection Benchmark')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    parser.add_argument('--image-set', type=str, help='Image set from wider_face to test')
    args = parser.parse_args()
    
    def haar_detector(image):
        from models.haarcascade.haarcascade_annote import get_bounding_boxes
        return get_bounding_boxes(image)
    
    def yunet_detector(image):
        from models.yunet.yunet_annote import get_bounding_boxes
        return get_bounding_boxes(image)
    
    # Read annotations
    wider_face_dir = 'benchmark/wider_face'
    annotations_file = os.path.join(wider_face_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
    all_annotations = []
    annotations = read_wider_annotations(annotations_file, args.image_set, debug=args.debug, )
    all_annotations.extend(annotations)
    
    if args.debug:
        print(f"\nTotal images in dataset: {len(all_annotations)}")
    
    print("\nEvaluating Haar Cascade detector...")
    haar_results = evaluate_detector(
        haar_detector, 
        all_annotations, 
        "Haar Cascade",
        debug=args.debug,
        max_images=args.max_images
    )
    
    print("\nEvaluating YuNet detector...")
    yunet_results = evaluate_detector(
        yunet_detector,
        all_annotations,
        "YuNet",
        debug=args.debug,
        max_images=args.max_images
    )
    
    print("\nHaar Cascade Results:")
    for k, v in haar_results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    print("\nYuNet Results:")
    for k, v in yunet_results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    main()
