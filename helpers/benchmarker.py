import cv2
import os
from pathlib import Path
import time
import argparse


def read_wider_annotations(annotation_file, debug=False):
    """Read Wider Face annotation file and return image paths and face locations."""
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

            if line.endswith('.jpg'):
                current_image = line
                face_count = int(lines[i + 1].strip())  # Next line is the face count
                faces = []

                for j in range(face_count):
                    values = list(map(float, lines[i + 2 + j].strip().split()))
                    x, y, w, h = values[:4]  # Only the first four values are the bounding box
                    faces.append([x, y, w, h])

                annotations.append({
                    'image_path': current_image,
                    'faces': faces
                })

                i += face_count + 2  # Skip to the next image entry
            else:
                i += 1

    if debug:
        print(f"Total annotations read: {len(annotations)}")

    return annotations

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
        image_path = os.path.join('benchmark/wider_face/WIDER_val/images', anno['image_path'])
        if debug:
            print(f"[DEBUG] Processing: {anno['image_path']}")
            
        if not os.path.exists(image_path):
            skipped_images += 1
            continue
            
        image = cv2.imread(image_path)
        if image is None:
            skipped_images += 1
            continue
            
        start_time = time.time()
        detected_faces = detector_func(image)
        end_time = time.time()
        total_time += end_time - start_time
        
        gt_faces = [face for face in anno['faces']]
        if debug:
            print(f'[DEBUG] detected_faces: {detected_faces}')
            print(f'[DEBUG] gt_faces: {gt_faces}')
        
        total_gt_faces += len(gt_faces)
        total_detected_faces += len(detected_faces)
        
        for gt_face in gt_faces:
            for det_face in detected_faces:
                iou = calculate_iou(gt_face, det_face)
                if debug:
                    print(f"[DEBUG] IOU: {iou}")
                if iou >= 0.5:
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

