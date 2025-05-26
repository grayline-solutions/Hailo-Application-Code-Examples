# object_detection_val.py
from pathlib import Path
import sys
import gc # For garbage collection, to get a slightly cleaner memory reading
# import psutil # For more accurate process memory, if available and desired later
from typing import List, Dict, Tuple, Any, Optional

import cv2
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from utils import IMAGE_EXTENSIONS


def get_approx_deep_size(obj, seen_ids=None):
    """
    Recursively estimates the approximate deep size of an object in bytes.
    Handles basic types, strings, dicts, lists, tuples, sets.
    Tries to avoid double counting and circular references.
    """
    if seen_ids is None:
        seen_ids = set()

    if id(obj) in seen_ids:
        return 0

    size = sys.getsizeof(obj)
    seen_ids.add(id(obj))

    if isinstance(obj, dict):
        size += sum(get_approx_deep_size(v, seen_ids) + get_approx_deep_size(k, seen_ids) for k, v in obj.items())
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_approx_deep_size(i, seen_ids) for i in obj)
    
    return size


def log_collection_memory_usage(variable_name: str, collection: Any, logger_obj):
    """Logs the number of items and approximate memory usage of a collection."""
    if collection is None:
        logger_obj.info(f"Memory for '{variable_name}': Collection is None.")
        return

    try:
        num_items = len(collection)
    except TypeError:
        num_items = "N/A (not a sized collection)"

    # Optional: Trigger garbage collection before measuring for a potentially cleaner number,
    # though this might impact performance if called very frequently.
    # gc.collect() 

    approx_size_bytes = get_approx_deep_size(collection)
    approx_size_mb = approx_size_bytes / (1024 * 1024)
    logger_obj.info(f"Collection '{variable_name}': Items = {num_items}, Approx. Memory = {approx_size_mb:.2f} MB")

    # If psutil is available, you can also log process memory:
    # try:
    #     import psutil
    #     process = psutil.Process(os.getpid())
    #     process_mem_mb = process.memory_info().rss / (1024 * 1024)
    #     logger_obj.info(f"Process Memory (RSS): {process_mem_mb:.2f} MB after populating '{variable_name}'")
    # except ImportError:
    #     pass # psutil not available


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
        logger.error(f"Data YAML ('{yaml_file_path}') must contain a non-empty '{split_key}' key.")
        return image_files_list, local_ground_truth_map
    
    dataset_root = Path(data_config.get('path', yaml_file_path.parent)).resolve()
    logger.info(f"Using dataset root: {dataset_root} for {split_key} split.")

    image_source_definition = data_config[split_key]
    collected_raw_image_paths: List[Path] = [] # Store Path objects first

    if isinstance(image_source_definition, str):
        source_path_str = image_source_definition.strip()
        if source_path_str.lower().endswith(".txt"):
            txt_file_path = _resolve_image_path(source_path_str, dataset_root, yaml_file_path.parent)
            if txt_file_path and txt_file_path.is_file():
                logger.info(f"Reading image list from: {txt_file_path}")
                with open(txt_file_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                for line_path in tqdm(lines, desc=f"Resolving images from {txt_file_path.name}", unit="path"):
                    resolved_img_path = _resolve_image_path(line_path, dataset_root, txt_file_path.parent)
                    if resolved_img_path and resolved_img_path.is_file():
                        collected_raw_image_paths.append(resolved_img_path)
                    else:
                        logger.warning(f"Path from {txt_file_path.name}: '{line_path}' -> '{resolved_img_path}' not found or not a file.")
            else:
                logger.error(f"Image list file not found: {txt_file_path} (from '{source_path_str}')")
        else: 
            single_image_dir = _resolve_image_path(source_path_str, dataset_root, yaml_file_path.parent)
            if single_image_dir and single_image_dir.is_dir():
                logger.info(f"Scanning for images in directory: {single_image_dir}")
                # Glob first to get a total for tqdm
                glob_paths = list(single_image_dir.rglob('*'))
                for item in tqdm(glob_paths, desc=f"Scanning {single_image_dir.name}", unit="file"):
                    if item.suffix.lower() in IMAGE_EXTENSIONS and item.is_file():
                        collected_raw_image_paths.append(item.resolve())
            else:
                logger.error(f"Image directory not found: {single_image_dir} (from '{source_path_str}')")
    elif isinstance(image_source_definition, list):
        for dir_idx, dir_path_str_item in enumerate(tqdm(image_source_definition, desc="Processing source directories", unit="dir")):
            if not isinstance(dir_path_str_item, str):
                logger.warning(f"Skipping non-string item in image directory list: {dir_path_str_item}")
                continue
            current_image_dir = _resolve_image_path(dir_path_str_item.strip(), dataset_root, yaml_file_path.parent)
            if current_image_dir and current_image_dir.is_dir():
                # Glob first for tqdm total
                glob_paths = list(current_image_dir.rglob('*'))
                for item in tqdm(glob_paths, desc=f"Scanning {current_image_dir.name}", unit="file", leave=False):
                    if item.suffix.lower() in IMAGE_EXTENSIONS and item.is_file():
                        collected_raw_image_paths.append(item.resolve())
            else:
                logger.warning(f"Image directory from list not found: {current_image_dir} (from '{dir_path_str_item}')")
    else:
        logger.error(f"Unsupported format for '{split_key}' in YAML: {type(image_source_definition)}.")
        return image_files_list, local_ground_truth_map

    if not collected_raw_image_paths:
        logger.error(f"No image paths were successfully collected for the '{split_key}' split.")
        return image_files_list, local_ground_truth_map
    
    # Remove duplicates that might arise from multiple listings/globs, preserving order somewhat
    unique_collected_image_paths = sorted(list(set(collected_raw_image_paths)), key=lambda p: str(p))
    if len(unique_collected_image_paths) < len(collected_raw_image_paths):
        logger.info(f"Removed {len(collected_raw_image_paths) - len(unique_collected_image_paths)} duplicate image paths.")
    
    log_collection_memory_usage("collected_image_paths (Path objects)", unique_collected_image_paths, logger)

    logger.info(f"Processing {len(unique_collected_image_paths)} unique image paths for labels and dimensions.")
    for img_path_obj in tqdm(unique_collected_image_paths, desc=f"Loading {split_key} GT data", unit="image"):
        abs_img_path_str = str(img_path_obj)
        img = cv2.imread(abs_img_path_str) # This loads the image to get dimensions
        if img is None:
            logger.warning(f"Could not read image {abs_img_path_str} during GT loading. Skipping.")
            continue
        h, w = img.shape[:2]
        image_files_list.append(abs_img_path_str) # Add to final list only if readable

        label_file = _derive_label_path(img_path_obj, dataset_root)
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
        logger.error(f"No images were successfully processed (e.g., all unreadable) for the '{split_key}' split.")
    
    logger.info(f"Finished GT loading. Found {len(image_files_list)} readable images.")
    log_collection_memory_usage(f"{split_key}_image_files_list (paths)", image_files_list, logger)
    log_collection_memory_usage(f"{split_key}_ground_truth_map (GT data)", local_ground_truth_map, logger)
    
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
    all_total_gt_instances_overall=0; all_total_images_with_any_gt_set=set()
    all_aps_50_list:List[float]=[]; all_aps_50_95_list:List[float]=[]
    overall_tp_sum_for_pr=0; overall_relevant_preds_sum_for_pr=0; overall_gt_sum_for_pr=0

    logger.info("Starting metrics calculation per class...")
    # --- TQDM for the main loop over classes ---
    for class_id in tqdm(range(num_classes), desc="Calculating Class Metrics", unit="class"):
        class_name = class_names[class_id]
        current_class_predictions: List[Dict[str, Any]] = []
        current_class_ground_truths: List[Dict[str, Any]] = []
        images_containing_gt_for_class_set = set()
        num_gt_instances_this_class = 0

        # Aggregate predictions and GTs for the current class
        # This loop iterates all accumulated predictions for *each class*.
        # If tqdm is desired here, ensure it's not too verbose.
        # It might be better to log memory *after* this aggregation for the class.
        for pred_item in hailo_preds_data: # This is iterating all_hailo_predictions
            img_path = pred_item['image_path']
            for det in pred_item['detections']:
                if det['class_id'] == class_id:
                    current_class_predictions.append(det)
            
            if img_path in gt_map:
                gt_image_info = gt_map[img_path]
                has_gt_in_this_image_for_class_this_time = False
                for gt_label in gt_image_info['labels']:
                    if gt_label['class_id'] is not None: all_total_images_with_any_gt_set.add(img_path)
                    if gt_label['class_id'] == class_id:
                        current_class_ground_truths.append(gt_label)
                        num_gt_instances_this_class += 1
                        has_gt_in_this_image_for_class_this_time = True
                if has_gt_in_this_image_for_class_this_time: images_containing_gt_for_class_set.add(img_path)
        
        # Log memory for per-class temporary collections (can be verbose, enable if needed)
        # log_collection_memory_usage(f"preds_cls_{class_name}", current_class_predictions, logger)
        # log_collection_memory_usage(f"GTs_cls_{class_name}", current_class_ground_truths, logger)
        
        all_total_gt_instances_overall += num_gt_instances_this_class
        p50, r50, ap50, map50_95_class = 0.0, 0.0, 0.0, 0.0
        
        if num_gt_instances_this_class == 0:
            p50 = 0.0 # No GTs, P=0 (as no TPs possible)
        else:
            ap50_calc, p50_calc, r50_calc, tp_this_class_for_pr = calculate_ap_for_class(
                current_class_predictions, current_class_ground_truths, 0.50
            )
            ap50, p50, r50 = ap50_calc, p50_calc, r50_calc
            
            overall_tp_sum_for_pr += tp_this_class_for_pr
            if p50 > 1e-9: 
                overall_relevant_preds_sum_for_pr += tp_this_class_for_pr / p50
            elif current_class_predictions: 
                overall_relevant_preds_sum_for_pr += len(current_class_predictions)
            overall_gt_sum_for_pr += num_gt_instances_this_class

            # --- TQDM for IoU threshold loop for mAP50-95 --- (can be nested, might be too much)
            # Let's keep it simple for now and not add tqdm here unless this specific part is identified as a huge bottleneck visually.
            aps_for_map50_95_range: List[float] = []
            # for iou_thresh_int in tqdm(range(50, 100, 5), desc=f"mAP50-95 IoUs for {class_name[:10]}..", leave=False, unit="IoU"):
            for iou_thresh_int in range(50, 100, 5): # No inner tqdm for now
                iou_val = iou_thresh_int / 100.0
                ap_at_iou, _, _, _ = calculate_ap_for_class(
                    current_class_predictions, current_class_ground_truths, iou_val
                )
                aps_for_map50_95_range.append(ap_at_iou)
            map50_95_class = np.mean(aps_for_map50_95_range) if aps_for_map50_95_range else 0.0
        
        class_summary_metrics.append({
            'name': class_name, 'images_gt': len(images_containing_gt_for_class_set),
            'instances_gt': num_gt_instances_this_class, 'P': p50, 'R': r50,
            'mAP50': ap50, 'mAP50-95': map50_95_class
        })
        
        if num_gt_instances_this_class > 0:
            all_aps_50_list.append(ap50)
            all_aps_50_95_list.append(map50_95_class)
            
    log_collection_memory_usage("class_summary_metrics (metrics list)", class_summary_metrics, logger)

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