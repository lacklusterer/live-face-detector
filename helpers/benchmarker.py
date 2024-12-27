import cv2
import numpy as np
import os
from pathlib import Path
import time
import argparse
import numpy as np

def read_fddb_annotations(annotation_file, debug=False):
    """Read FDDB annotation file and return image paths and face locations."""
    annotations = []
    current_image = None
    face_count = 0
    
    if debug:
        print(f"Reading annotations from: {annotation_file}")
    
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('200'):
                current_image = line
                face_count = int(lines[i + 1])
                faces = []
                
                for j in range(face_count):
                    values = list(map(float, lines[i + 2 + j].strip().split()))
                    faces.append(values)
                
                annotations.append({
                    'image_path': current_image,
                    'faces': faces
                })
                
                i += face_count + 2
            else:
                i += 1
    
    if debug:
        print(f"Total annotations read: {len(annotations)}")
    
    return annotations

def ellipse_to_bbox(ellipse):
    """Convert elliptical annotation to rectangular bbox."""
    major_axis = ellipse[0]
    minor_axis = ellipse[1]
    angle = ellipse[2]
    center_x = ellipse[3]
    center_y = ellipse[4]
    
    width = 2 * major_axis
    height = 2 * minor_axis
    
    x1 = int(center_x - width/2)
    y1 = int(center_y - height/2)
    x2 = int(center_x + width/2)
    y2 = int(center_y + height/2)
    
    return [x1, y1, x2-x1, y2-y1]

def calculate_iou(box1, box2):
    """Calculate intersection over union between two bboxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_detector(detector_func, annotations, detector_name="Unknown", debug=False, max_images=None):
    """Evaluate face detector performance."""
    total_gt_faces = 0
    total_detected_faces = 0
    total_correct_detections = 0
    total_time = 0
    processed_images = 0
    skipped_images = 0
    
    for anno in annotations[:max_images] if max_images else annotations:
        image_path = os.path.join('benchmark', anno['image_path']) + '.jpg'
        if debug:
            print(f"Processing: {anno['image_path']}")
            
        if not os.path.exists(image_path):
            skipped_images += 1
            continue
            
        image = cv2.imread(image_path)
        if image is None:
            skipped_images += 1
            continue
            
        start_time = time.time()
        detected_image = detector_func(image)
        end_time = time.time()
        total_time += end_time - start_time
        
        if isinstance(detected_image, tuple):
            detected_faces = detected_image[1]
        else:
            diff = cv2.absdiff(image, detected_image)
            mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_faces = [cv2.boundingRect(cnt) for cnt in contours]
        
        gt_faces = [ellipse_to_bbox(face) for face in anno['faces']]
        print(f'[DEBUG] {gt_faces}')
        
        total_gt_faces += len(gt_faces)
        total_detected_faces += len(detected_faces)
        
        for gt_face in gt_faces:
            for det_face in detected_faces:
                if calculate_iou(gt_face, det_face) >= 0.5:
                    total_correct_detections += 1
                    break
        
        processed_images += 1
    
    precision = total_correct_detections / total_detected_faces if total_detected_faces > 0 else 0
    recall = total_correct_detections / total_gt_faces if total_gt_faces > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_time = total_time / processed_images if processed_images > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_time': avg_time,
        'total_images': processed_images,
        'skipped_images': skipped_images,
        'total_gt_faces': total_gt_faces,
        'total_detected_faces': total_detected_faces,
        'total_correct_detections': total_correct_detections,
        'total_time': total_time
    }

def main():
    parser = argparse.ArgumentParser(description='Face Detection Benchmark')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to process')
    args = parser.parse_args()
    
    def haar_detector(image):
        from models.yunet.yunet_annote import get_annote
        return get_annote(image)
    
    def yunet_detector(image):
        from models.haarcascade.haarcascades import process
        return process(image)
    
    # Read annotations
    fold_dir = 'benchmark/FDDB-folds'
    all_annotations = []
    for i in range(1, 11):
        annotation_file = os.path.join(fold_dir, f'FDDB-fold-{i:02d}-ellipseList.txt')
        annotations = read_fddb_annotations(annotation_file, debug=args.debug)
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
