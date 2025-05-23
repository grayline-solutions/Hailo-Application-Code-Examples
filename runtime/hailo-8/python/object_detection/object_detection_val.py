# object_detection_val.py
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import yaml
from loguru import logger

from utils import IMAGE_EXTENSIONS

def _resolve_image_path(path_str: str, primary_root: Path, secondary_root: Optional[Path] = None) -> Optional[Path]:
    """Helper to resolve image paths, trying primary_root then optionally secondary_root."""
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve() if p.exists() else None
    
    resolved_path = (primary_root / p).resolve()
    if resolved_path.exists():
        return resolved_path
    
    if secondary_root:
        resolved_path_secondary = (secondary_root / p).resolve()
        if resolved_path_secondary.exists():
            return resolved_path_secondary
            
    return None


def _derive_label_path(image_path: Path, dataset_root_for_label_search: Path) -> Path:
    """
    Derives the corresponding label file path from an image file path.
    Assumes a common structure like 'images/' -> 'labels/'.
    Tries to find 'images' segment relative to dataset_root and replace it.
    If not found, tries replacing the immediate parent directory if it's 'images'.
    Fallback if no 'images' segment, tries putting 'labels' as sibling to image's parent folder.
    """
    img_path_str = str(image_path)
    
    # Try to find 'images' relative to a known dataset root for more reliable replacement
    try:
        relative_to_root = image_path.relative_to(dataset_root_for_label_search)
        parts = list(relative_to_root.parts)
        # Try to find and replace 'images' or 'Images'
        for i, p in enumerate(parts):
            if p.lower() == 'images':
                parts[i] = 'labels'
                label_rel_path = Path(*parts[:-1]) / (image_path.stem + '.txt')
                return (dataset_root_for_label_search / label_rel_path).resolve()
    except ValueError: # image_path is not under dataset_root_for_label_search
        pass # Proceed to more general replacement

    # More general replacement: replace the last occurrence of "images" in the path
    # This is more heuristic.
    if 'images' in img_path_str.lower():
        # Attempt to replace the last 'images' path component
        parts = list(image_path.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].lower() == 'images':
                label_parts = parts[:i] + ['labels'] + parts[i+1:]
                label_path = Path(*label_parts[:-1]) / (image_path.stem + '.txt') # up to parent, then stem.txt
                return label_path.resolve()
    
    # Fallback: assume labels are in a parallel directory structure to the image's direct parent
    # e.g., if image is in '.../split_name/images_subfolder/img.jpg', try '.../split_name/labels_subfolder/img.txt'
    # This is very heuristic. A common structure is often dataset/images/split and dataset/labels/split
    # So if image is dataset/something/split/img.jpg, label might be dataset/labels/split/img.txt
    # Let's assume the parent of image_path.parent is where 'images' and 'labels' might be siblings.
    # if image_path.parent.name.lower() == 'images':
    #    label_dir = image_path.parent.parent / 'labels'
    # else:
    #    label_dir = image_path.parent.parent / 'labels' / image_path.parent.name
    # This heuristic is getting complicated. The primary method should be replacing 'images' with 'labels'.
    # A simpler robust fallback if 'images' isn't in path: assume labels are next to image's parent dir
    # e.g. parent/images_dir/img.jpg -> parent/labels_dir/img.txt
    # For now, rely on the 'images' replacement. If that fails, users might need to ensure their structure matches.
    # A simpler fallback: assume labels dir is sibling to image dir's parent, with same name as image dir's parent
    # e.g. path/to/DATASET/subset/images -> path/to/DATASET/subset/labels
    # This means label_file = image_path.parent.with_name('labels') / (image_path.stem + '.txt')
    # This is too simple. The "replace 'images' with 'labels'" is the most common.

    # Final simple fallback: look for a "labels" folder parallel to the image's folder
    label_path = image_path.parent.parent / "labels" / image_path.parent.name / (image_path.stem + '.txt')
    if not label_path.exists(): # One more common: if image is dataset/images/split/img.jpg -> dataset/labels/split/img.txt
        label_path = dataset_root_for_label_search / "labels" / image_path.relative_to(dataset_root_for_label_search).parent.name / (image_path.stem + ".txt")
        if not label_path.parent.is_dir(): # Check if intermediate "labels" and split name dir exists
             # If image in dataset/images_A/img.jpg -> expect dataset/labels_A/img.txt
             label_path = image_path.parent.with_name(image_path.parent.name.lower().replace("images", "labels")) / (image_path.stem + ".txt")


    logger.debug(f"Derived label path for {image_path} as {label_path}")
    return label_path


def load_ground_truth_data(
    data_yaml_path: str, use_test_split: bool
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Loads ground truth data from the specified data.yaml file.
    Returns a list of image file paths for the validation/test set
    and a dictionary mapping image paths to their ground truth annotations.
    """
    # This map is local and will be returned.
    local_ground_truth_map: Dict[str, Dict[str, Any]] = {}
    image_files_list: List[str] = []

    yaml_file_path = Path(data_yaml_path)
    if not yaml_file_path.is_file():
        logger.error(f"Data YAML file not found: {data_yaml_path}")
        return image_files_list, local_ground_truth_map

    try:
        with open(yaml_file_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading or parsing YAML file '{yaml_file_path}': {e}")
        return image_files_list, local_ground_truth_map

    split_key = 'test' if use_test_split else 'val'
    if split_key not in data_config or not data_config[split_key]:
        logger.error(f"Data YAML ('{yaml_file_path}') must contain a non-empty '{split_key}' key specifying the {split_key} image source(s).")
        return image_files_list, local_ground_truth_map
    
    # Dataset root: 'path' in YAML, or YAML's parent directory
    dataset_root = Path(data_config.get('path', yaml_file_path.parent)).resolve()
    logger.info(f"Using dataset root: {dataset_root}")

    image_source_definition = data_config[split_key]
    collected_image_paths: List[Path] = []

    if isinstance(image_source_definition, str):
        source_path_str = image_source_definition.strip()
        if source_path_str.lower().endswith(".txt"):
            txt_file_path = _resolve_image_path(source_path_str, dataset_root, yaml_file_path.parent)
            if txt_file_path and txt_file_path.is_file():
                logger.info(f"Reading image list from: {txt_file_path}")
                with open(txt_file_path, 'r') as f:
                    for line in f:
                        img_path_in_txt = line.strip()
                        if not img_path_in_txt: continue
                        resolved_img_path = _resolve_image_path(img_path_in_txt, dataset_root, txt_file_path.parent)
                        if resolved_img_path and resolved_img_path.is_file():
                            collected_image_paths.append(resolved_img_path)
                        else:
                            logger.warning(f"Image path from {txt_file_path.name}: '{img_path_in_txt}' not found or not a file (tried resolving against {dataset_root} and {txt_file_path.parent}).")
            else:
                logger.error(f"Image list file not found or not a file: {source_path_str} (resolved to {txt_file_path})")
        else: # Single directory
            single_image_dir = _resolve_image_path(source_path_str, dataset_root, yaml_file_path.parent)
            if single_image_dir and single_image_dir.is_dir():
                logger.info(f"Scanning for images in directory: {single_image_dir}")
                for item in sorted(single_image_dir.rglob('*')):
                    if item.suffix.lower() in IMAGE_EXTENSIONS:
                        collected_image_paths.append(item.resolve())
            else:
                logger.error(f"Image directory not found or not a directory: {source_path_str} (resolved to {single_image_dir})")
    elif isinstance(image_source_definition, list): # List of directories
        for dir_path_str_item in image_source_definition:
            if not isinstance(dir_path_str_item, str):
                logger.warning(f"Skipping non-string item in image directory list: {dir_path_str_item}")
                continue
            current_image_dir = _resolve_image_path(dir_path_str_item.strip(), dataset_root, yaml_file_path.parent)
            if current_image_dir and current_image_dir.is_dir():
                logger.info(f"Scanning for images in directory: {current_image_dir}")
                for item in sorted(current_image_dir.rglob('*')):
                    if item.suffix.lower() in IMAGE_EXTENSIONS:
                        collected_image_paths.append(item.resolve())
            else:
                logger.warning(f"Image directory from list not found or not a directory: {dir_path_str_item} (resolved to {current_image_dir})")
    else:
        logger.error(f"Unsupported format for '{split_key}' images in YAML: {type(image_source_definition)}. Expected str or list.")
        return image_files_list, local_ground_truth_map

    if not collected_image_paths:
        logger.error(f"No image paths were successfully collected for the '{split_key}' split.")
        return image_files_list, local_ground_truth_map

    logger.info(f"Collected {len(collected_image_paths)} image paths for '{split_key}' split. Processing for labels and dimensions.")
    
    for img_path_obj in collected_image_paths:
        abs_img_path_str = str(img_path_obj)
        img = cv2.imread(abs_img_path_str)
        if img is None:
            logger.warning(f"Could not read image {abs_img_path_str}. Skipping.")
            continue
        h, w = img.shape[:2]
        image_files_list.append(abs_img_path_str)

        label_file = _derive_label_path(img_path_obj, dataset_root) # Use dataset_root as a base for label search context
        
        current_image_gts: Dict[str, Any] = {'labels': [], 'width': w, 'height': h}
        if label_file.exists() and label_file.is_file():
            with open(label_file, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw / 2) * w; y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w; y2 = (cy + bh / 2) * h
                            current_image_gts['labels'].append({'class_id': class_id, 'bbox_abs': [x1, y1, x2, y2]})
                        except ValueError:
                            logger.warning(f"Skipping malformed line in label file {label_file}: '{line.strip()}'")
        # else:
            # logger.debug(f"Label file not found for {abs_img_path_str} at expected location {label_file}")

        local_ground_truth_map[abs_img_path_str] = current_image_gts
            
    if not image_files_list:
        logger.error(f"No images were successfully processed for the '{split_key}' split (e.g., all unreadable).")
    
    logger.info(f"Finished processing. Found {len(image_files_list)} readable images and {len(local_ground_truth_map)} ground truth entries for '{split_key}' split.")
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