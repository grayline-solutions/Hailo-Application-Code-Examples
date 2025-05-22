# object_detection_val.py
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import yaml
from loguru import logger

from utils import IMAGE_EXTENSIONS

def load_ground_truth_data(
    data_yaml_path: str, use_test_split: bool
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Loads ground truth data from the specified data.yaml file.
    Returns a list of image file paths for the validation/test set
    and a dictionary mapping image paths to their ground truth annotations.
    """
    # This map is now local and will be returned.
    local_ground_truth_map: Dict[str, Dict[str, Any]] = {}
    image_files_list: List[str] = [] # Renamed from val_image_files for clarity

    if not data_yaml_path: # Should be caught by arg parsing, but good to have
        logger.warning("No data.yaml path provided to load_ground_truth_data.")
        return image_files_list, local_ground_truth_map

    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading or parsing YAML file '{data_yaml_path}': {e}")
        return image_files_list, local_ground_truth_map

    split_key = 'test' if use_test_split else 'val'
    labels_split_key = 'test_labels' if use_test_split else 'val_labels'

    if split_key not in data_config:
        logger.error(f"Data YAML ('{data_yaml_path}') must contain a '{split_key}' key specifying the {split_key} image directory.")
        return image_files_list, local_ground_truth_map
    
    yaml_parent = Path(data_yaml_path).parent
    dataset_root = Path(data_config.get('path', yaml_parent)).resolve()
    
    image_dir_rel_path = data_config[split_key]
    image_dir = (dataset_root / image_dir_rel_path).resolve()

    label_dir: Optional[Path] = None
    label_dir_rel_options = []

    # Try explicit label path first if provided for the current split
    if labels_split_key in data_config and data_config[labels_split_key]:
         label_dir_rel_options.append(Path(data_config[labels_split_key]))
    
    # Common pattern: 'labels' directory relative to 'images' directory or dataset structure
    # Option 1: replace 'images' with 'labels' in the image directory path
    # e.g., if image_dir_rel_path is 'path/to/project/images/val2017', try 'path/to/project/labels/val2017'
    if "images" in str(image_dir_rel_path):
        label_dir_rel_options.append(Path(str(image_dir_rel_path).replace("images", "labels", 1)))
    
    # Option 2: 'labels' directory as a sibling to the image directory's parent folder, with the same split name
    # e.g., if image_dir is '../datasets/coco/images/val', try '../datasets/coco/labels/val'
    label_dir_rel_options.append(Path(image_dir_rel_path).parent / "labels" / Path(image_dir_rel_path).name)

    # Option 3: 'labels' directory directly under dataset_root, with the same split name
    # e.g., if image_dir_rel_path is 'images/val', try 'labels/val' from dataset_root
    label_dir_rel_options.append(Path("labels") / Path(image_dir_rel_path).name)


    for rel_path_option in label_dir_rel_options:
        potential_label_dir = (dataset_root / rel_path_option).resolve()
        if potential_label_dir.exists() and potential_label_dir.is_dir():
            label_dir = potential_label_dir
            break
    
    if not label_dir:
        logger.error(f"Could not automatically determine or find label directory for images in {image_dir}.")
        logger.info(f"Attempted relative label paths from dataset root '{dataset_root}': {[str(opt) for opt in label_dir_rel_options]}")
        return image_files_list, local_ground_truth_map

    logger.info(f"Loading {split_key} images from: {image_dir}")
    logger.info(f"Expecting labels in: {label_dir}")

    for img_file_path_obj in sorted(image_dir.rglob('*')):
        if img_file_path_obj.suffix.lower() in IMAGE_EXTENSIONS:
            abs_img_path_str = str(img_file_path_obj.resolve())
            
            img = cv2.imread(abs_img_path_str)
            if img is None:
                logger.warning(f"Could not read image {abs_img_path_str}. Skipping.")
                continue
            h, w = img.shape[:2]
            image_files_list.append(abs_img_path_str) 

            label_file = label_dir / (img_file_path_obj.stem + '.txt')
            current_image_gts: Dict[str, Any] = {'labels': [], 'width': w, 'height': h}
            if label_file.exists():
                with open(label_file, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])
                                x1 = (cx - bw / 2) * w
                                y1 = (cy - bh / 2) * h
                                x2 = (cx + bw / 2) * w
                                y2 = (cy + bh / 2) * h
                                current_image_gts['labels'].append({'class_id': class_id, 'bbox_abs': [x1, y1, x2, y2]})
                            except ValueError:
                                logger.warning(f"Skipping malformed line in label file {label_file}: '{line.strip()}'")
            local_ground_truth_map[abs_img_path_str] = current_image_gts
            
    if not image_files_list:
        logger.error(f"No {split_key} images found or readable in {image_dir} or its subdirectories.")
    
    logger.info(f"Processed {len(image_files_list)} images and {len(local_ground_truth_map)} ground truth entries for {split_key} split.")

    return image_files_list, local_ground_truth_map


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """ Calculate IoU between two boxes (x1, y1, x2, y2) """
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    inter_width = max(0, x2_i - x1_i)
    inter_height = max(0, y2_i - y1_i)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_ap_for_class(
    predictions_for_class: List[Dict[str, Any]], 
    ground_truths_for_class: List[Dict[str, Any]], 
    iou_threshold: float
) -> Tuple[float, float, float, int]:
    """
    Calculates Average Precision (AP), Precision, and Recall for a single class.
    Returns: AP, Precision, Recall, number of True Positives
    """
    sorted_preds = sorted(predictions_for_class, key=lambda x: x['score'], reverse=True)
    num_gt = len(ground_truths_for_class)

    if num_gt == 0:
        precision_val = 0.0 # If no GTs, P=0 (no TPs possible, even if preds exist making them FPs)
        return 0.0, precision_val, 0.0, 0

    gt_used_flags = [False] * num_gt
    tp_list = np.zeros(len(sorted_preds))
    fp_list = np.zeros(len(sorted_preds))

    if not sorted_preds: # No predictions for this class (but GTs exist)
        return 0.0, 0.0, 0.0, 0 # AP=0, P usually undefined/0, R=0

    for i, pred in enumerate(sorted_preds):
        pred_box = pred['box_abs_xyxy']
        best_iou = -1.0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truths_for_class):
            if gt_used_flags[j]:
                continue
            iou = calculate_iou(pred_box, gt['bbox_abs'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            if not gt_used_flags[best_gt_idx]:
                tp_list[i] = 1
                gt_used_flags[best_gt_idx] = True
            else:
                fp_list[i] = 1 
        else:
            fp_list[i] = 1
            
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    recalls = tp_cumsum / num_gt # num_gt is > 0 here
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9) 
    
    precisions_ap = np.concatenate(([0.], precisions, [0.]))
    recalls_ap = np.concatenate(([0.], recalls, [1.]))    
    
    for k in range(len(precisions_ap) - 2, -1, -1):
        precisions_ap[k] = np.maximum(precisions_ap[k], precisions_ap[k+1])
        
    indices = np.where(recalls_ap[:-1] != recalls_ap[1:])[0] + 1
    ap = np.sum((recalls_ap[indices] - recalls_ap[indices-1]) * precisions_ap[indices])
    
    total_tp = np.sum(tp_list)
    total_fp = np.sum(fp_list) # Total FPs for this class at all confidences
    
    # P, R for the table should reflect overall performance for the class based on AP calc logic
    overall_precision = total_tp / (total_tp + total_fp + 1e-9)
    overall_recall = total_tp / (num_gt + 1e-9) # num_gt is > 0
    
    return ap, overall_precision, overall_recall, int(total_tp)


def calculate_and_print_metrics_table(
    hailo_preds_data: List[Dict[str, Any]], 
    gt_map: Dict[str, Dict[str, Any]], 
    class_names: List[str]
) -> None:
    num_classes = len(class_names)
    class_summary_metrics: List[Dict[str, Any]] = [] 

    all_total_gt_instances_overall = 0
    all_total_images_with_any_gt_set = set()
    
    all_aps_50_list: List[float] = []
    all_aps_50_95_list: List[float] = []
    
    # For more accurate 'all' P/R: sum TPs, (TPs+FPs), and GTs across all classes
    overall_tp_sum_for_pr = 0
    overall_relevant_preds_sum_for_pr = 0 # This is sum of (TP+FP) for each class for P calculation
    overall_gt_sum_for_pr = 0

    for class_id in range(num_classes):
        class_name = class_names[class_id]
        
        current_class_predictions: List[Dict[str, Any]] = []
        current_class_ground_truths: List[Dict[str, Any]] = []
        
        images_containing_gt_for_class_set = set()
        num_gt_instances_this_class = 0

        for pred_item in hailo_preds_data:
            img_path = pred_item['image_path']
            for det in pred_item['detections']:
                if det['class_id'] == class_id:
                    current_class_predictions.append(det)
            
            if img_path in gt_map:
                gt_image_info = gt_map[img_path]
                has_gt_in_this_image_for_class_this_time = False
                for gt_label in gt_image_info['labels']:
                    if gt_label['class_id'] is not None: # Any GT for this image
                        all_total_images_with_any_gt_set.add(img_path)
                    if gt_label['class_id'] == class_id:
                        current_class_ground_truths.append(gt_label)
                        num_gt_instances_this_class += 1
                        has_gt_in_this_image_for_class_this_time = True
                if has_gt_in_this_image_for_class_this_time:
                    images_containing_gt_for_class_set.add(img_path)
        
        all_total_gt_instances_overall += num_gt_instances_this_class

        p50, r50, ap50, map50_95_class = 0.0, 0.0, 0.0, 0.0
        
        if num_gt_instances_this_class == 0:
            # If predictions exist, P=0. If no preds, P=0 (or 1, but 0 for consistency if instances=0).
            p50 = 0.0 if current_class_predictions else 0.0 
        else:
            ap50_calc, p50_calc, r50_calc, tp_this_class_for_pr = calculate_ap_for_class(
                current_class_predictions, current_class_ground_truths, 0.50
            )
            ap50, p50, r50 = ap50_calc, p50_calc, r50_calc
            
            overall_tp_sum_for_pr += tp_this_class_for_pr
            if p50 > 1e-9: # Calculate (TP+FP) for this class
                overall_relevant_preds_sum_for_pr += tp_this_class_for_pr / p50
            elif current_class_predictions: # If P=0 but predictions exist, they are all FPs
                overall_relevant_preds_sum_for_pr += len(current_class_predictions)
            overall_gt_sum_for_pr += num_gt_instances_this_class

            aps_for_map50_95_range: List[float] = []
            for iou_thresh_int in range(50, 100, 5):
                iou_val = iou_thresh_int / 100.0
                ap_at_iou, _, _, _ = calculate_ap_for_class(
                    current_class_predictions, current_class_ground_truths, iou_val
                )
                aps_for_map50_95_range.append(ap_at_iou)
            map50_95_class = np.mean(aps_for_map50_95_range) if aps_for_map50_95_range else 0.0
        
        class_summary_metrics.append({
            'name': class_name,
            'images_gt': len(images_containing_gt_for_class_set),
            'instances_gt': num_gt_instances_this_class,
            'P': p50, 'R': r50, 'mAP50': ap50, 'mAP50-95': map50_95_class
        })
        
        if num_gt_instances_this_class > 0:
            all_aps_50_list.append(ap50)
            all_aps_50_95_list.append(map50_95_class)

    num_total_images_with_any_gt = len(all_total_images_with_any_gt_set)
    
    all_p_overall = overall_tp_sum_for_pr / (overall_relevant_preds_sum_for_pr + 1e-9) if overall_relevant_preds_sum_for_pr > 0 else 0.0
    all_r_overall = overall_tp_sum_for_pr / (overall_gt_sum_for_pr + 1e-9) if overall_gt_sum_for_pr > 0 else 0.0
    all_map50_avg = np.mean(all_aps_50_list) if all_aps_50_list else 0.0
    all_map50_95_avg = np.mean(all_aps_50_95_list) if all_aps_50_95_list else 0.0

    header_format = "{:<20} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10}"
    # Using logger.info for table output for consistency if logs are captured
    logger.info("\nValidation Metrics:")
    logger.info(header_format.format("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)"))
    separator = "-" * (20 + 7 + 10 + 10 + 10 + 10 + 10 + (6 * 1))
    logger.info(separator)
    
    logger.info(header_format.format(
        "all", 
        num_total_images_with_any_gt, 
        all_total_gt_instances_overall,
        f"{all_p_overall:.3f}", 
        f"{all_r_overall:.3f}", 
        f"{all_map50_avg:.3f}", 
        f"{all_map50_95_avg:.3f}"
    ))

    for metrics in class_summary_metrics:
        logger.info(header_format.format(
            metrics['name'], 
            metrics['images_gt'], 
            metrics['instances_gt'],
            f"{metrics['P']:.3f}", 
            f"{metrics['R']:.3f}", 
            f"{metrics['mAP50']:.3f}", 
            f"{metrics['mAP50-95']:.3f}"
        ))
    logger.info(separator)