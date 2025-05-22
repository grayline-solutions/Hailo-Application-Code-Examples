#!/usr/bin/env python3
"""
Object Detection Script for Hailo HEF Models with Optional Validation Metrics.

This script performs object detection using a pre-trained model in Hailo's
HEF format on an accelerator. It supports two main modes of operation:

1. Normal Inference Mode:
   - Activated when the `--data` argument (for a YAML dataset file) is NOT provided.
   - Processes input from various sources specified by the `--input` argument:
     - A single image file.
     - A directory of image files.
     - A video file.
     - A live camera stream (e.g., by passing "camera").
   - Class names for display and visualization are sourced from the text file
     specified by the `--labels` argument (e.g., "coco.txt"). Each line in this
     file should contain one class name, in the order corresponding to the
     model's output indices.
   - Output: Displays detections on screen and can optionally save output images
     (for image inputs) or a video file (for stream inputs via `--save_stream_output`).

2. Validation Metrics Mode:
   - Activated when the `--data` argument IS provided, pointing to a YOLO-style
     `data.yaml` file. This YAML defines the dataset, including paths to image
     splits and class name information.
   - In this mode, the `--input` argument is generally IGNORED. The script will
     process images from the dataset split defined in the `data.yaml`.
   - By default, it processes the 'val' (validation) split. Use the `--test_split`
     flag to process the 'test' split instead.
   - Accumulates inference outputs and calculates COCO-style metrics (P, R, mAP50,
     mAP50-95), displaying them in a table similar to `yolo detect val`.

   - **Critical: Class Name Handling for Validation Metrics:**
     - The `names` list within the `data.yaml` file (specified by `--data`) is
       considered the **AUTHORITATIVE SOURCE** for class names and their order.
       These names are used for calculating and displaying the metrics table.
       This YAML should align with how the model was trained and how the ground
       truth label files (which use integer class IDs) were generated.
     - The `--labels` argument (path to a `.txt` file with class names, one per line)
       is STILL REQUIRED.
       - If the `data.yaml` file lacks a `names` list, the script will FALL BACK to
         using the class names from the `--labels` `.txt` file for metrics.
         A warning will be issued, and you must ensure this `.txt` file accurately
         reflects the model's class order.
       - If both `data.yaml` (with a `names` list) and the `--labels` `.txt` file
         are provided, the script will use the YAML names for the metrics table.
         It will issue a warning if these two sources seem to differ significantly
         (e.g., in class count or names), but the YAML names will take precedence
         for metrics.
       - The internal `ObjectDetectionUtils` (used for preprocessing and potentially
         drawing bounding box labels if visualizations are active) is initialized
         using the class names from the `--labels` `.txt` file. If these names
         differ from the authoritative YAML names used for metrics, visualizations
         on detected boxes might display names from the `.txt` file, while the
         metrics table uses names from the YAML. For pure metrics generation, this
         visual discrepancy is less critical than the accuracy of the metrics table itself.

Key Arguments:
  -n, --net: Path to the HEF model file.
  -i, --input: Input source (image, folder, video, "camera"). Ignored if --data is used.
  -l, --labels: Path to a text file with class names (one per line).
  -b, --batch_size: Number of images per batch for inference.
  --data: Path to the data.yaml file to enable validation metrics mode.
  --test_split: If --data is used, use the 'test' split instead of 'val'.
  -s, --save_stream_output: If processing a stream, save the output to a video file.

Example Usage:
  # Normal inference on an image
  python object_detection.py -n model.hef -i image.jpg -l coco.txt

  # Validation metrics using 'val' split from dataset.yaml
  python object_detection.py -n model.hef --data dataset.yaml -l coco.txt

  # Validation metrics using 'test' split
  python object_detection.py -n model.hef --data dataset.yaml -l coco.txt --test_split
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import queue
import threading
import cv2
from typing import List, Dict, Optional, Tuple, Any
import yaml # For data.yaml

from object_detection_utils import ObjectDetectionUtils

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_images_opencv, validate_images, divide_list_to_batches, IMAGE_EXTENSIONS


CAMERA_CAP_WIDTH = 1920
CAMERA_CAP_HEIGHT = 1080

# Global list to store all predictions from Hailo for metrics calculation
all_hailo_predictions: List[Dict[str, Any]] = []
# Global dict to store ground truth data, keyed by image path
# Value: {'labels': [{'class_id': 0, 'bbox_abs': [x1,y1,x2,y2]}, ...], 'width': w, 'height': h}
ground_truth_map: Dict[str, Dict[str, Any]] = {}


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example with Validation Metrics")
    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        default="yolov7.hef"
    )
    parser.add_argument(
        "-i", "--input",
        default="zidane.jpg",
        help="Path to the input - either an image, a folder of images, 'camera', or video file. Ignored if --data is used for validation."
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-l", "--labels",
        default="coco.txt",
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
    )
    parser.add_argument(
        "-s", "--save_stream_output",
        action="store_true",
        help="Save the output of the inference from a stream."
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to data.yaml file (Ultralytics format). Enables validation metrics mode.",
        default=None
    )
    parser.add_argument(
        "--test_split",
        action="store_true",
        help="Use the 'test' split from data.yaml for validation instead of 'val'."
    )
    args = parser.parse_args()

    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    if args.data and not os.path.exists(args.data):
        raise FileNotFoundError(f"Data YAML file not found: {args.data}")

    return args


def load_ground_truth_data(data_yaml_path: str, use_test_split: bool) -> List[str]:
    """
    Loads ground truth data from the specified data.yaml file.
    Populates global `ground_truth_map`.
    Returns a list of image file paths for the validation or test set.
    """
    global ground_truth_map
    ground_truth_map = {}

    if not data_yaml_path:
        logger.warning("No data.yaml provided. Skipping ground truth loading.")
        return []

    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    split_key = 'test' if use_test_split else 'val'
    labels_split_key = 'test_labels' if use_test_split else 'val_labels' # For explicit label paths

    if split_key not in data_config:
        logger.error(f"Data YAML must contain a '{split_key}' key specifying the {split_key} image directory.")
        return []
    
    yaml_parent = Path(data_yaml_path).parent
    dataset_root = Path(data_config.get('path', yaml_parent)).resolve()
    
    image_dir_rel = data_config[split_key]
    image_dir = (dataset_root / image_dir_rel).resolve()

    label_dir_rel_options = [
        Path(str(image_dir_rel).replace("images", "labels", 1)),
        Path(image_dir_rel).parent / "labels" / Path(image_dir_rel).name,
        Path("labels") / Path(image_dir_rel).name
    ]
    if labels_split_key in data_config: # Explicit label path for the chosen split
         label_dir_rel_options.insert(0, Path(data_config[labels_split_key]))

    label_dir = None
    for rel_path in label_dir_rel_options:
        potential_label_dir = (dataset_root / rel_path).resolve()
        if potential_label_dir.exists() and potential_label_dir.is_dir():
            label_dir = potential_label_dir
            break
    
    if not label_dir:
        logger.error(f"Could not automatically determine or find label directory for images in {image_dir}. Tried options based on: {label_dir_rel_options}")
        return []

    logger.info(f"Loading validation images from: {image_dir}")
    logger.info(f"Expecting labels in: {label_dir}")

    val_image_files: List[str] = []
    for img_file_path in sorted(image_dir.rglob('*')): # rglob for nested structures
        if img_file_path.suffix.lower() in IMAGE_EXTENSIONS:
            abs_img_path_str = str(img_file_path.resolve())
            val_image_files.append(abs_img_path_str)
            
            img = cv2.imread(abs_img_path_str)
            if img is None:
                logger.warning(f"Could not read image {abs_img_path_str}. Skipping.")
                continue
            h, w = img.shape[:2]

            label_file = label_dir / (img_file_path.stem + '.txt')
            current_image_gts: Dict[str, Any] = {'labels': [], 'width': w, 'height': h}
            if label_file.exists():
                with open(label_file, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5: # class_id + 4 box coords
                            class_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            current_image_gts['labels'].append({'class_id': class_id, 'bbox_abs': [x1, y1, x2, y2]})
            ground_truth_map[abs_img_path_str] = current_image_gts
            
    if not val_image_files:
        logger.error(f"No validation images found in {image_dir} or its subdirectories.")
    if not ground_truth_map and val_image_files:
        logger.warning(f"No corresponding labels found in {label_dir}. Metrics will be affected.")

    return val_image_files


def preprocess_from_image_list(
    images_with_paths: List[Dict[str, Any]],
    batch_size: int,
    input_queue: queue.Queue,
    model_input_width: int,
    model_input_height: int,
    utils: ObjectDetectionUtils
) -> None:
    """ Process a list of images (with paths) and enqueue them. """
    for batch_meta in divide_list_to_batches(images_with_paths, batch_size):
        original_frames_with_ids: List[Dict[str, Any]] = []
        processed_frames: List[np.ndarray] = []
        for item in batch_meta:
            original_frames_with_ids.append({'path': item['path'], 'frame': item['cv_image']})
            # Preprocess expects BGR, converts to RGB if needed by model
            # Assuming utils.preprocess handles RGB conversion if necessary
            processed_frame = utils.preprocess(item['cv_image'], model_input_width, model_input_height)
            processed_frames.append(processed_frame)
        
        input_queue.put((original_frames_with_ids, processed_frames))


def preprocess(
    image_paths: List[str], 
    cap: Optional[cv2.VideoCapture],
    batch_size: int,
    input_queue: queue.Queue,
    model_net_width: int, # Renamed for clarity
    model_net_height: int, # Renamed for clarity
    utils: ObjectDetectionUtils
) -> None:
    if cap is None: # Processing image files
        images_to_load_meta = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images_to_load_meta.append({'path': img_path, 'cv_image': img})
            else:
                logger.warning(f"Failed to load image: {img_path}, skipping.")
        
        if not images_to_load_meta:
            logger.error("No images could be loaded from the provided paths.")
            input_queue.put(None) # Signal end if no images
            return
        preprocess_from_image_list(images_to_load_meta, batch_size, input_queue, model_net_width, model_net_height, utils)
    else: # Processing camera or video stream
        # Simplified for brevity: metrics typically not for live streams in this context
        # preprocess_from_cap would need to be adapted to yield image paths/IDs if metrics were desired here
        logger.info("Processing from stream. Metrics accumulation is not enabled for streams in this script.")
        # Original preprocess_from_cap logic (ensure it sends data in compatible format if used with metrics later)
        frames = []
        processed_frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # For streams, create a dummy path/ID
            dummy_path = f"stream_frame_{frame_idx}" 
            frame_idx +=1

            original_meta = {'path': dummy_path, 'frame': frame.copy()} # Send copy
            processed_frame = utils.preprocess(frame, model_net_width, model_net_height) # Assuming BGR input to preprocess

            # Batching for stream
            frames.append(original_meta)
            processed_frames.append(processed_frame)

            if len(frames) == batch_size:
                input_queue.put(([f.copy() for f in frames], [pf.copy() for pf in processed_frames])) # Send copies
                frames, processed_frames = [], []
        
        if frames: # remaining frames
            input_queue.put(([f.copy() for f in frames], [pf.copy() for pf in processed_frames]))


    input_queue.put(None)  # Add sentinel value to signal end of input


def postprocess(
    output_queue: queue.Queue,
    cap: Optional[cv2.VideoCapture],
    save_stream_output: bool,
    utils: ObjectDetectionUtils,
    run_validation: bool,
    model_net_height: int, # Network input height
    model_net_width: int   # Network input width
) -> None:
    global all_hailo_predictions
    image_counter = 0 # Generic counter
    output_path_obj = Path('output') # Define once

    # Video writer setup (if needed)
    out_video_writer = None
    # Ensure output directory exists for images if not streaming
    if cap is None and not run_validation: # Only make dir if saving individual images
        output_path_obj.mkdir(exist_ok=True)

    while True:
        result_item = output_queue.get()
        if result_item is None:
            break

        original_frame_meta, infer_results = result_item 
        # original_frame_meta is now {'path': str, 'frame': np.ndarray} (from callback)
        
        original_cv_image = original_frame_meta['frame']
        image_path = original_frame_meta.get('path', f"processed_item_{image_counter}")
        image_counter += 1

        # Handle HailoRT version differences in output structure
        if isinstance(infer_results, list) and len(infer_results) == 1 and \
           isinstance(infer_results[0], (list, np.ndarray)): # Heuristic for older HailoRT output
            infer_results = infer_results[0]
        
        # Extract detections. Threshold is low to get most predictions for mAP.
        # Assumes `extract_detections` returns boxes normalized 0-1 relative to model input.
        raw_detections_on_model_input = utils.extract_detections(infer_results, threshold=0.001)

        if run_validation and image_path in ground_truth_map: # Only process if GT exists (for validation)
            img_h, img_w = original_cv_image.shape[:2]
            
            # Calculate scale and padding applied during preprocess
            # This mirrors the logic in ObjectDetectionUtils.preprocess
            scale_ratio = min(model_net_width / img_w, model_net_height / img_h)
            new_scaled_img_w = int(img_w * scale_ratio)
            new_scaled_img_h = int(img_h * scale_ratio)
            pad_x = (model_net_width - new_scaled_img_w) // 2
            pad_y = (model_net_height - new_scaled_img_h) // 2

            current_image_pred_list = []
            for i in range(raw_detections_on_model_input['num_detections']):
                # Box from extract_detections is [ymin, xmin, ymax, xmax], normalized 0-1 on network input.
                norm_ymin, norm_xmin, norm_ymax, norm_xmax = raw_detections_on_model_input['detection_boxes'][i]
                
                # Scale to absolute pixel coordinates on the (padded) model input
                abs_xmin_pad = norm_xmin * model_net_width
                abs_ymin_pad = norm_ymin * model_net_height
                abs_xmax_pad = norm_xmax * model_net_width
                abs_ymax_pad = norm_ymax * model_net_height

                # Transform to original image coordinates by removing padding and scaling
                x1_orig = (abs_xmin_pad - pad_x) / scale_ratio
                y1_orig = (abs_ymin_pad - pad_y) / scale_ratio
                x2_orig = (abs_xmax_pad - pad_x) / scale_ratio
                y2_orig = (abs_ymax_pad - pad_y) / scale_ratio
                
                # Clip to original image boundaries and ensure x1<x2, y1<y2
                final_box_abs = [
                    np.clip(min(x1_orig, x2_orig), 0, img_w -1), # Use img_w-1, img_h-1 as max coord
                    np.clip(min(y1_orig, y2_orig), 0, img_h -1),
                    np.clip(max(x1_orig, x2_orig), 0, img_w -1),
                    np.clip(max(y1_orig, y2_orig), 0, img_h -1)
                ]

                # Filter out zero-area boxes after clipping and transformation
                if final_box_abs[2] > final_box_abs[0] and final_box_abs[3] > final_box_abs[1]:
                    current_image_pred_list.append({
                        'box_abs_xyxy': final_box_abs, # [x1, y1, x2, y2] absolute
                        'score': raw_detections_on_model_input['detection_scores'][i],
                        'class_id': raw_detections_on_model_input['detection_classes'][i]
                    })
            
            all_hailo_predictions.append({
                'image_path': image_path,
                'width': img_w,
                'height': img_h,
                'detections': current_image_pred_list
            })

        # Visualization (optional, or if not in pure validation mode)
        # For simplicity, visualization is less emphasized when run_validation is True.
        # You might want to add a flag like --show-output for validation.
        show_output_anyway = cap is not None # Show if it's a stream

        if show_output_anyway or (not run_validation and not save_stream_output) :
             # `draw_detections` expects `image` and `detections`
             # `detections` should be the output of `extract_detections` (normalized boxes)
            frame_with_detections = utils.draw_detections(
                raw_detections_on_model_input, original_cv_image.copy() # Pass copy to draw on
            )
            if cap is not None:
                cv2.imshow("Output", frame_with_detections)
                if save_stream_output:
                    if out_video_writer is None: # Initialize video writer
                        output_path_obj.mkdir(exist_ok=True)
                        frame_h_vid, frame_w_vid = frame_with_detections.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        fps_vid = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20.0
                        out_video_writer = cv2.VideoWriter(str(output_path_obj / 'output_video.avi'), fourcc, fps_vid, (frame_w_vid, frame_h_vid))
                    out_video_writer.write(frame_with_detections)

            else: # Not a capture stream, saving individual image
                 if not run_validation: # Only save if not in validation mode (or add specific flag)
                    cv2.imwrite(str(output_path_obj / f"output_{Path(image_path).stem}.png"), frame_with_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if cap: cap.release()
            if out_video_writer: out_video_writer.release()
            cv2.destroyAllWindows()
            # This break will stop postprocessing early, might miss some items in queue.
            # Consider a softer exit mechanism if breaking mid-queue is an issue.
            # For now, this is standard.
            logger.info("User quit.")
            # To ensure clean exit, we might need to signal other threads.
            # For this script structure, output_queue.put(None) is handled by infer's main logic.
            break 
            
    if out_video_writer:
        out_video_writer.release()
    cv2.destroyAllWindows()
    output_queue.task_done()


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
) -> Tuple[float, float, float, float]: # Returns AP, Precision, Recall, num_tp
    """
    Calculates Average Precision (AP), Precision, and Recall for a single class.
    Precision/Recall are calculated based on all detections for this class using the AP logic.
    """
    # Sort predictions by score descending
    sorted_preds = sorted(predictions_for_class, key=lambda x: x['score'], reverse=True)
    
    num_gt = len(ground_truths_for_class)
    if num_gt == 0: # No ground truths for this class
        # If predictions exist, they are all FPs, so P=0. If no predictions, P might be 0 (no TPs).
        # R=0, AP=0.
        precision_val = 0.0 # Always 0 if no GTs, as TP will be 0.
        return (0.0, precision_val, 0.0, 0) # AP, Precision, Recall, num_tp

    # Mark all GTs as not used for matching yet for this AP calculation
    gt_used_flags = [False] * num_gt

    tp_list = np.zeros(len(sorted_preds)) # True Positive status for each prediction
    fp_list = np.zeros(len(sorted_preds)) # False Positive status for each prediction

    if not sorted_preds: # No predictions for this class
        return (0.0, 1.0, 0.0, 0) # AP=0, P=1 (no false positives), R=0 (all GTs are FNs)

    for i, pred in enumerate(sorted_preds):
        pred_box = pred['box_abs_xyxy']
        best_iou = -1.0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truths_for_class):
            if gt_used_flags[j]: # If this GT already matched by a higher-score prediction
                continue
            
            iou = calculate_iou(pred_box, gt['bbox_abs'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold:
            if not gt_used_flags[best_gt_idx]: # Check again, just in case (should be redundant here)
                tp_list[i] = 1
                gt_used_flags[best_gt_idx] = True
            else: # Matched a GT that was already "taken" by a higher score prediction (should not happen if logic is right for single pass)
                  # This case means the GT was already matched by a previous (higher score) pred; this one is an FP
                fp_list[i] = 1
        else: # No GT match or IoU too low
            fp_list[i] = 1
            
    # Compute precision and recall arrays
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    recalls = tp_cumsum / num_gt if num_gt > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9) # Add epsilon to avoid division by zero
    
    # Standard COCO AP calculation (area under PR curve with all points)
    precisions_ap = np.concatenate(([0.], precisions, [0.])) # Pad for AP calculation
    recalls_ap = np.concatenate(([0.], recalls, [1.]))    # Pad for AP calculation
    
    # Ensure precision is monotonically decreasing for AP calculation
    for k in range(len(precisions_ap) - 2, -1, -1):
        precisions_ap[k] = np.maximum(precisions_ap[k], precisions_ap[k+1])
        
    # Find indices where recall changes to compute area
    indices = np.where(recalls_ap[:-1] != recalls_ap[1:])[0] + 1
    ap = np.sum((recalls_ap[indices] - recalls_ap[indices-1]) * precisions_ap[indices])
    
    # For the table P,R values: use the P/R at the last TP point, or average?
    # Ultralytics often reports P/R values at a specific confidence or optimal F1.
    # Here, let's use overall P/R from the full list of detections for this class.
    total_tp = np.sum(tp_list)
    total_fp = np.sum(fp_list)
    
    overall_precision = total_tp / (total_tp + total_fp + 1e-9)
    overall_recall = total_tp / (num_gt + 1e-9) if num_gt > 0 else 0.0
    
    return ap, overall_precision, overall_recall, int(total_tp)


def calculate_and_print_metrics_table(
    hailo_preds_data: List[Dict[str, Any]], 
    gt_map: Dict[str, Dict[str, Any]], 
    class_names: List[str]
):
    num_classes = len(class_names)
    
    # Metrics storage for each class
    # {'name', 'images_gt', 'instances_gt', 'P', 'R', 'mAP50', 'mAP50-95'}
    class_summary_metrics: List[Dict[str, Any]] = [] 

    # For overall "all" statistics
    all_total_gt_instances = 0
    all_total_images_with_gt_set = set()
    all_aps_50 = []
    all_aps_50_95 = []
    all_total_tp_p_r = 0 # For a global P/R, sum of TPs at some conditions
    all_total_preds_p_r = 0 # Sum of predictions considered for P/R

    for class_id in range(num_classes):
        class_name = class_names[class_id]
        
        current_class_predictions: List[Dict[str, Any]] = []
        current_class_ground_truths: List[Dict[str, Any]] = []
        
        images_containing_gt_for_class = set()
        num_gt_instances_for_class = 0

        # Corrected iteration:
        for prediction_item in hailo_preds_data: # Iterate through the list of dictionaries
            img_path = prediction_item['image_path'] # Access 'image_path'
            
            # Add predictions for this class from this image
            # 'detections' is a key in prediction_item dictionary
            for det in prediction_item['detections']: 
                if det['class_id'] == class_id:
                    current_class_predictions.append(det) 
            
            if img_path in gt_map:
                gt_image_info = gt_map[img_path]
                has_gt_in_this_image_for_class = False
                for gt_label in gt_image_info['labels']:
                    if gt_label['class_id'] == class_id:
                        current_class_ground_truths.append(gt_label)
                        num_gt_instances_for_class +=1
                        has_gt_in_this_image_for_class = True
                if has_gt_in_this_image_for_class:
                    images_containing_gt_for_class.add(img_path)
                # Ensure all_total_images_with_gt_set is populated correctly based on GT presence
                if gt_image_info['labels']: # If the image has any GT labels at all
                    all_total_images_with_gt_set.add(img_path)

        # ... (rest of the metrics calculation for the class) ...
        all_total_gt_instances += num_gt_instances_for_class

        if num_gt_instances_for_class == 0: # If no GT instances for this class
            p50, r50, ap50, map50_95_class = 0.0, 0.0, 0.0, 0.0
        else:
            # Calculate AP@0.50 and associated P, R
            ap50, p50, r50, _ = calculate_ap_for_class(current_class_predictions, current_class_ground_truths, 0.50)
            
            # Calculate mAP50-95
            aps_for_map50_95_range: List[float] = []
            for iou_thresh_int in range(50, 100, 5):
                iou_val = iou_thresh_int / 100.0
                ap_at_iou, _, _, _ = calculate_ap_for_class(current_class_predictions, current_class_ground_truths, iou_val)
                aps_for_map50_95_range.append(ap_at_iou)
            map50_95_class = np.mean(aps_for_map50_95_range) if aps_for_map50_95_range else 0.0

        class_summary_metrics.append({
            'name': class_name,
            'images_gt': len(images_containing_gt_for_class),
            'instances_gt': num_gt_instances_for_class,
            'P': p50, # Precision from AP@0.5 calc
            'R': r50, # Recall from AP@0.5 calc
            'mAP50': ap50,
            'mAP50-95': map50_95_class
        })
        
        if num_gt_instances_for_class > 0: # Only include in 'all' averages if class had GT instances
            all_aps_50.append(ap50)
            all_aps_50_95.append(map50_95_class)

    # Calculate 'all' class summary metrics
    # P and R for 'all' are typically calculated globally, not averaged.
    # For simplicity here, we average the per-class P,R and mAPs for classes with instances.
    # A more accurate 'all' P/R would require re-running TP/FP counts across all classes combined.
    
    # Calculate 'all' P, R, mAP50, mAP50-95
    # For 'all' P and R, it's better to calculate from total TPs and FPs across all predictions/GTs
    # This is a simplification:
    all_p_avg = np.mean([m['P'] for m in class_summary_metrics if m['instances_gt'] > 0]) if any(m['instances_gt'] > 0 for m in class_summary_metrics) else 0.0
    all_r_avg = np.mean([m['R'] for m in class_summary_metrics if m['instances_gt'] > 0]) if any(m['instances_gt'] > 0 for m in class_summary_metrics) else 0.0
    all_map50_avg = np.mean(all_aps_50) if all_aps_50 else 0.0
    all_map50_95_avg = np.mean(all_aps_50_95) if all_aps_50_95 else 0.0

    # Print the table header
    header_format = "{:<20} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print("\nValidation Metrics:")
    print(header_format.format("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)"))
    print("-" * (20 + 7 + 10 + 10 + 10 + 10 + 10 + (6*1))) # Adjust length for spacing

    # Print 'all' row
    # Total images with any GT label:
    num_total_images_with_any_gt = len(all_total_images_with_gt_set)
    
    print(header_format.format(
        "all", 
        num_total_images_with_any_gt, 
        all_total_gt_instances,
        f"{all_p_avg:.3f}", 
        f"{all_r_avg:.3f}", 
        f"{all_map50_avg:.3f}", 
        f"{all_map50_95_avg:.3f}"
    ))

    # Print per-class rows
    for metrics in class_summary_metrics:
        print(header_format.format(
            metrics['name'], 
            metrics['images_gt'], 
            metrics['instances_gt'],
            f"{metrics['P']:.3f}", 
            f"{metrics['R']:.3f}", 
            f"{metrics['mAP50']:.3f}", 
            f"{metrics['mAP50-95']:.3f}"
        ))
    print("-" * (20 + 7 + 10 + 10 + 10 + 10 + 10 + (6*1)))


def infer(
    input_source: str,
    save_stream_output: bool,
    net_path: str,
    labels_txt_path: str, # Renamed for clarity, this is from args.labels
    batch_size: int,
    data_yaml_path: Optional[str] = None, # From args.data
    use_test_split: bool = False # From args.test_split
) -> None:
    global all_hailo_predictions, ground_truth_map # Clear for new run
    all_hailo_predictions = []
    ground_truth_map = {} # Populated by load_ground_truth_data

    # This will be the definitive list of class names for the current run.
    authoritative_class_names: Optional[List[str]] = None
    run_validation_metrics = False

    # Initialize image_file_paths_for_inference to an empty list
    image_file_paths_for_inference: List[str] = [] 
    cap: Optional[cv2.VideoCapture] = None # Initialize cap as well

    # 1. Initialize ObjectDetectionUtils with the provided --labels TXT file.
    # This instance will be used for preprocessing, drawing, etc.
    # Its internal .labels list comes from this TXT file.
    det_utils = ObjectDetectionUtils(labels_txt_path)
    class_names_from_txt_file: List[str] = list(det_utils.labels) # Get a copy

    if not class_names_from_txt_file:
        logger.error(f"CRITICAL: The --labels file ('{labels_txt_path}') is empty or could not be loaded. Cannot determine class names.")
        return

    # 2. Determine authoritative_class_names and if validation should run
    if data_yaml_path:
        logger.info(f"Data YAML provided: {data_yaml_path}. Attempting to load validation set.")
        # load_ground_truth_data populates global ground_truth_map
        image_file_paths_for_inference = load_ground_truth_data(data_yaml_path, use_test_split)

        if image_file_paths_for_inference: # ground_truth_map is implicitly checked by this
            run_validation_metrics = True
            logger.info(f"Validation mode active: {len(image_file_paths_for_inference)} images to process.")

            parsed_yaml_names: Optional[List[str]] = None
            num_classes_yaml: Optional[int] = None
            try:
                with open(data_yaml_path, 'r') as f_yaml:
                    yaml_cfg = yaml.safe_load(f_yaml)
                    if 'names' in yaml_cfg and isinstance(yaml_cfg['names'], list):
                        parsed_yaml_names = yaml_cfg['names']
                    if 'nc' in yaml_cfg and isinstance(yaml_cfg['nc'], int):
                        num_classes_yaml = yaml_cfg['nc']
            except Exception as e:
                logger.warning(f"Could not effectively parse data.yaml ('{data_yaml_path}') for class names/nc: {e}")

            if parsed_yaml_names:
                authoritative_class_names = parsed_yaml_names
                logger.info(f"Using class names from data.yaml ('{data_yaml_path}') as authoritative for metrics: {len(authoritative_class_names)} classes.")
                # (Optional consistency checks and warnings as before)
                if num_classes_yaml and num_classes_yaml != len(parsed_yaml_names):
                    logger.warning(f"YAML 'nc' ({num_classes_yaml}) mismatches 'names' list length ({len(parsed_yaml_names)}). Trusting 'names' list length from YAML.")
                if set(class_names_from_txt_file) != set(parsed_yaml_names) or len(class_names_from_txt_file) != len(parsed_yaml_names):
                    logger.warning(
                        f"Class names from --labels file ('{labels_txt_path}') differ from authoritative names in data.yaml ('{data_yaml_path}'). "
                        f"YAML names will be used for metrics table. Visualizations will use names from '{labels_txt_path}'."
                    )
            elif class_names_from_txt_file: # Fallback if YAML has no names
                authoritative_class_names = class_names_from_txt_file
                logger.warning(
                    f"data.yaml ('{data_yaml_path}') does not contain a 'names' list. "
                    f"Falling back to class names from --labels file ('{labels_txt_path}') as authoritative for metrics. "
                    "Ensure this is consistent with your model and ground truth IDs."
                )
            else: 
                logger.error(f"CRITICAL: No class names available from data.yaml or '{labels_txt_path}'. Cannot proceed with validation accurately.")
                run_validation_metrics = False
        else: # image_file_paths_for_inference was empty after load_ground_truth_data
            logger.warning(f"Failed to load images/labels for validation from '{data_yaml_path}'. Disabling metrics mode.")
            run_validation_metrics = False
            # image_file_paths_for_inference is already [] if load_ground_truth_data returned empty

    # 3. Setup for normal inference if not in validation mode or if validation setup failed
    if not run_validation_metrics:
        logger.info(f"Normal inference mode. Using input: {input_source}")
        authoritative_class_names = class_names_from_txt_file # Use TXT names for normal mode
        logger.info(f"Using class names from --labels file: {labels_txt_path}") # Already logged if TXT is the source

        # Initialize image_file_paths_for_inference and cap for normal mode
        # This part is crucial and was correctly placed in your original snippet
        if input_source.lower() == "camera":
            temp_cap = cv2.VideoCapture(0) # Attempt to create
            if not temp_cap.isOpened():   # Check if successfully opened IMMEDIATELY
                logger.error("Cannot open camera")
                if temp_cap: temp_cap.release() # Good practice to release if created
                return
            # Now we know temp_cap is opened and valid
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)
            cap = temp_cap # Assign to the broader scoped 'cap'
            image_file_paths_for_inference = [] 
        elif Path(input_source).is_file() and Path(input_source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            temp_cap = cv2.VideoCapture(input_source) # Attempt to create
            if not temp_cap.isOpened():  # Check if successfully opened IMMEDIATELY
                logger.error(f"Cannot open video file: {input_source}")
                if temp_cap: temp_cap.release()
                return
            # Now we know temp_cap is opened and valid
            cap = temp_cap # Assign to the broader scoped 'cap'
            image_file_paths_for_inference = []
        else: 
            # In this case (image or folder input), cap remains None, which is intended.
            # No methods are called on 'cap' if it's None here.
            input_path_obj = Path(input_source)
            temp_image_paths = [] 
            if input_path_obj.is_file() and input_path_obj.suffix.lower() in IMAGE_EXTENSIONS:
                temp_image_paths = [str(input_path_obj.resolve())]
            elif input_path_obj.is_dir():
                temp_image_paths = sorted([str(p.resolve()) for p in input_path_obj.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
            
            if not temp_image_paths:
                logger.error(f"No valid images found for input: {input_source}")
                return
            image_file_paths_for_inference = temp_image_paths

            if batch_size > 1 and len(image_file_paths_for_inference) % batch_size != 0 :
                 logger.error(f"Number of images ({len(image_file_paths_for_inference)}) must be divisible by batch_size ({batch_size}) when batch_size > 1 for non-validation file-based runs.")
                 return

    if not authoritative_class_names: # Final check
        logger.error("CRITICAL: Authoritative class names could not be established. Exiting.")
        return

    # 4. Proceed with inference setup
    q_multiplier = 5 
    input_q_size = max(batch_size * q_multiplier, q_multiplier * 2)
    output_q_size = max(batch_size * q_multiplier, q_multiplier * 2)
    input_queue: queue.Queue = queue.Queue(maxsize=input_q_size)
    output_queue: queue.Queue = queue.Queue(maxsize=output_q_size)

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size, send_original_frame=True
    )
    network_input_shape = hailo_inference.get_input_shape()
    model_net_h, model_net_w = network_input_shape[0], network_input_shape[1]

    # det_utils (initialized from labels_txt_path) is used for threads
    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(image_file_paths_for_inference, cap, batch_size, input_queue, model_net_w, model_net_h, det_utils)
    )
    postprocess_thread = threading.Thread(
        target=postprocess,
        args=(output_queue, cap, save_stream_output, det_utils, run_validation_metrics, model_net_h, model_net_w)
    )

    preprocess_thread.start()
    postprocess_thread.start()

    hailo_inference.run() 
    
    preprocess_thread.join()
    output_queue.put(None)
    postprocess_thread.join()

    # 5. Calculate and print metrics if in validation mode
    if run_validation_metrics: # authoritative_class_names should be set if run_validation_metrics is True
        if all_hailo_predictions and ground_truth_map: # ground_truth_map check might be redundant if run_validation_metrics implies it's populated
            logger.info("Calculating validation metrics...")
            calculate_and_print_metrics_table(all_hailo_predictions, ground_truth_map, authoritative_class_names)
        else:
            logger.warning("Not enough data for validation metrics (e.g., no predictions accumulated or ground truth map empty despite validation mode).")

    logger.info('Inference/Validation run completed.')


def main() -> None:
    args = parse_args() # Ensure parse_args includes --test_split
    # Global lists/dicts are cleared/reinitialized within infer or load_ground_truth_data
    infer(
        input_source=args.input,
        save_stream_output=args.save_stream_output,
        net_path=args.net,
        labels_txt_path=args.labels, # Pass the path to the .txt labels file
        batch_size=args.batch_size,
        data_yaml_path=args.data,
        use_test_split=args.test_split # Pass the boolean flag
)


if __name__ == "__main__":
    main()