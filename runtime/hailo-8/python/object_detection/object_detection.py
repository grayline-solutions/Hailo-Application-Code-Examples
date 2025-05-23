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
from typing import List, Dict, Optional, Tuple, Any # Keep all necessary types
import yaml 

# Import the ObjectDetectionUtils class from object_detection_utils
from object_detection_utils import ObjectDetectionUtils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the HailoAsyncInference class and other utility functions
from utils import HailoAsyncInference, load_images_opencv, validate_images, divide_list_to_batches, IMAGE_EXTENSIONS

# Import the validation functions
from object_detection_val import (
    load_ground_truth_data,
    calculate_and_print_metrics_table
)

CAMERA_CAP_WIDTH = 1920
CAMERA_CAP_HEIGHT = 1080

# Global list to store all predictions from Hailo for metrics calculation
all_hailo_predictions: List[Dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
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
        help="Save annotated output. For image inputs, saves annotated images. For video/camera streams, saves an annotated video."
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


def preprocess_from_image_list(
    images_with_paths: List[Dict[str, Any]],
    batch_size: int,
    input_queue: queue.Queue,
    model_input_width: int,
    model_input_height: int,
    utils: ObjectDetectionUtils
) -> None:
    for batch_meta in divide_list_to_batches(images_with_paths, batch_size):
        original_frames_with_ids: List[Dict[str, Any]] = []
        processed_frames: List[np.ndarray] = []
        for item in batch_meta:
            original_frames_with_ids.append({'path': item['path'], 'frame': item['cv_image']})
            processed_frame = utils.preprocess(item['cv_image'], model_input_width, model_input_height)
            processed_frames.append(processed_frame)
        input_queue.put((original_frames_with_ids, processed_frames))

def preprocess(
    image_paths: List[str], 
    cap: Optional[cv2.VideoCapture],
    batch_size: int,
    input_queue: queue.Queue,
    model_net_width: int, 
    model_net_height: int, 
    utils: ObjectDetectionUtils
) -> None:
    if cap is None: 
        images_to_load_meta = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images_to_load_meta.append({'path': img_path, 'cv_image': img})
            else:
                logger.warning(f"Failed to load image: {img_path}, skipping.")
        
        if not images_to_load_meta:
            logger.error("No images could be loaded from the provided paths.")
            input_queue.put(None) 
            return
        preprocess_from_image_list(images_to_load_meta, batch_size, input_queue, model_net_width, model_net_height, utils)
    else: 
        logger.info("Processing from stream. Metrics accumulation is not enabled for streams in this script.")
        frames = []
        processed_frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dummy_path = f"stream_frame_{frame_idx}" 
            frame_idx +=1
            original_meta = {'path': dummy_path, 'frame': frame.copy()}
            processed_frame = utils.preprocess(frame, model_net_width, model_net_height)
            frames.append(original_meta)
            processed_frames.append(processed_frame)
            if len(frames) == batch_size:
                input_queue.put(([f.copy() for f in frames], [pf.copy() for pf in processed_frames]))
                frames, processed_frames = [], []
        if frames:
            input_queue.put(([f.copy() for f in frames], [pf.copy() for pf in processed_frames]))
    input_queue.put(None)


# Populates the global `all_hailo_predictions` if run_validation is True
# and appends the current image's predictions to it.
# This is used for validation metrics calculation.
# The function also handles visualization and saving output.
# It draws the detections on the original image and saves it if the save_output_flag is True.
# If the input is a stream, it saves the output as a video file.
# If the input is an image, it saves the output as an image file.
# The function also handles the cleanup of resources after processing.
# The function is designed to run in a separate thread.
# It continuously checks the output queue for results and processes them.
# The function will exit when it receives a None value in the output queue.
# The function also handles the display of the output in a window.
# It uses OpenCV to show the output in a window named "Output".
# The function will exit if the user presses 'q'.
def postprocess(
    output_queue: queue.Queue,
    cap: Optional[cv2.VideoCapture],
    save_output_flag: bool, # This will be args.save_stream_output, now treated as a general save flag
    utils: ObjectDetectionUtils,
    run_validation: bool, # This purely controls metrics data accumulation
    model_net_height: int,
    model_net_width: int
) -> None:
    global all_hailo_predictions
    image_counter = 0
    output_path_obj = Path('output')
    out_video_writer: Optional[cv2.VideoWriter] = None # Type hint for clarity

    while True:
        result_item = output_queue.get()
        if result_item is None:
            break

        original_frame_meta, infer_results = result_item
        original_cv_image = original_frame_meta['frame']
        image_path = original_frame_meta.get('path', f"processed_item_{image_counter}")
        image_counter += 1

        if isinstance(infer_results, list) and len(infer_results) == 1 and \
           isinstance(infer_results[0], (list, np.ndarray)):
            infer_results = infer_results[0]
        
        raw_detections_on_model_input = utils.extract_detections(infer_results, threshold=0.001)

        # --- 1. Accumulate predictions if in validation mode ---
        # This is independent of saving output.
        if run_validation:
            img_h, img_w = original_cv_image.shape[:2]
            scale_ratio = min(model_net_width / img_w, model_net_height / img_h) if img_w > 0 and img_h > 0 else 1.0
            new_scaled_img_w = int(img_w * scale_ratio)
            new_scaled_img_h = int(img_h * scale_ratio)
            pad_x = (model_net_width - new_scaled_img_w) // 2
            pad_y = (model_net_height - new_scaled_img_h) // 2
            current_image_pred_list = []
            for i in range(raw_detections_on_model_input['num_detections']):
                norm_ymin, norm_xmin, norm_ymax, norm_xmax = raw_detections_on_model_input['detection_boxes'][i]
                abs_xmin_pad = norm_xmin * model_net_width
                abs_ymin_pad = norm_ymin * model_net_height
                abs_xmax_pad = norm_xmax * model_net_width
                abs_ymax_pad = norm_ymax * model_net_height
                x1_orig = (abs_xmin_pad - pad_x) / scale_ratio if scale_ratio != 0 else 0
                y1_orig = (abs_ymin_pad - pad_y) / scale_ratio if scale_ratio != 0 else 0
                x2_orig = (abs_xmax_pad - pad_x) / scale_ratio if scale_ratio != 0 else 0
                y2_orig = (abs_ymax_pad - pad_y) / scale_ratio if scale_ratio != 0 else 0
                final_box_abs = [
                    np.clip(min(x1_orig, x2_orig), 0, img_w -1 if img_w > 0 else 0),
                    np.clip(min(y1_orig, y2_orig), 0, img_h -1 if img_h > 0 else 0),
                    np.clip(max(x1_orig, x2_orig), 0, img_w -1 if img_w > 0 else 0),
                    np.clip(max(y1_orig, y2_orig), 0, img_h -1 if img_h > 0 else 0)
                ]
                if final_box_abs[2] > final_box_abs[0] and final_box_abs[3] > final_box_abs[1]:
                    current_image_pred_list.append({
                        'box_abs_xyxy': final_box_abs,
                        'score': raw_detections_on_model_input['detection_scores'][i],
                        'class_id': raw_detections_on_model_input['detection_classes'][i]
                    })
            all_hailo_predictions.append({
                'image_path': image_path, 'width': img_w, 'height': img_h,
                'detections': current_image_pred_list
            })

        # --- 2. Handle visualization (drawing) and saving output ---
        # We need to draw detections if:
        #   a) We are saving the output (save_output_flag is True).
        #   b) It's a live stream (cap is not None), because we always imshow for streams.
        frame_to_process: Optional[np.ndarray] = None # To hold the annotated frame
        if save_output_flag or (cap is not None):
            frame_to_process = utils.draw_detections(
                raw_detections_on_model_input, original_cv_image.copy()
            )

        # Display if it's a live stream and we have a frame
        if cap is not None and frame_to_process is not None:
            cv2.imshow("Output", frame_to_process)

        # Save the output if the save flag is enabled and we have a frame
        if save_output_flag and frame_to_process is not None:
            output_path_obj.mkdir(exist_ok=True) # Ensure output directory exists

            if cap is not None: # Stream input: save to video
                if out_video_writer is None:
                    frame_h_vid, frame_w_vid = frame_to_process.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps_val = cap.get(cv2.CAP_PROP_FPS)
                    fps_vid = fps_val if fps_val and fps_val > 0 else 20.0 # Handle potential None from cap.get
                    video_output_path = str(output_path_obj / 'output_video.avi')
                    out_video_writer = cv2.VideoWriter(video_output_path, fourcc, fps_vid, (frame_w_vid, frame_h_vid))
                    logger.info(f"Saving output video to: {video_output_path}")
                
                if out_video_writer: # Check if writer was successfully initialized
                    out_video_writer.write(frame_to_process)
            else: # Image input (cap is None): save as individual image
                image_output_path = str(output_path_obj / f"output_{Path(image_path).stem}.png")
                cv2.imwrite(image_output_path, frame_to_process)
                # logger.info(f"Saved annotated image to: {image_output_path}") # Optional: log saved images

        # --- 3. UI handling for quit ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if cap: cap.release()
            if out_video_writer: out_video_writer.release()
            cv2.destroyAllWindows()
            logger.info("User quit.")
            break
            
    # Cleanup after loop
    if out_video_writer:
        out_video_writer.release()
    cv2.destroyAllWindows()
    output_queue.task_done()


def infer(
    input_source: str,
    save_stream_output: bool,
    net_path: str,
    labels_txt_path: str, 
    batch_size: int,
    data_yaml_path: Optional[str] = None,
    use_test_split: bool = False
) -> None:
    global all_hailo_predictions # Clear for new run
    all_hailo_predictions = [] 
    
    # This will hold the ground truth map if validation is active
    current_run_ground_truth_map: Dict[str, Dict[str, Any]] = {} 

    authoritative_class_names: Optional[List[str]] = None
    run_validation_metrics = False
    image_file_paths_for_inference: List[str] = [] 
    cap: Optional[cv2.VideoCapture] = None

    det_utils = ObjectDetectionUtils(labels_txt_path) # Initialized once
    class_names_from_txt_file: List[str] = list(det_utils.labels)

    if not class_names_from_txt_file:
        logger.error(f"CRITICAL: The --labels file ('{labels_txt_path}') is empty or could not be loaded.")
        return

    if data_yaml_path:
        logger.info(f"Data YAML provided: {data_yaml_path}. Attempting to load dataset for validation.")
        # Call imported function; it returns image paths and the ground truth map
        loaded_image_paths, loaded_gt_map = load_ground_truth_data(data_yaml_path, use_test_split)
        
        if loaded_image_paths: 
            image_file_paths_for_inference = loaded_image_paths
            current_run_ground_truth_map = loaded_gt_map # Store returned map
            run_validation_metrics = True
            logger.info(f"Validation mode active: {len(image_file_paths_for_inference)} images to process.")

            # Determine authoritative_class_names (logic from previous response)
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
                if num_classes_yaml and num_classes_yaml != len(parsed_yaml_names):
                    logger.warning(f"YAML 'nc' ({num_classes_yaml}) mismatches 'names' list length ({len(parsed_yaml_names)}). Trusting 'names' list length from YAML.")
                if set(class_names_from_txt_file) != set(parsed_yaml_names) or len(class_names_from_txt_file) != len(parsed_yaml_names):
                    logger.warning(
                        f"Class names from --labels file ('{labels_txt_path}') differ from authoritative names in data.yaml ('{data_yaml_path}'). "
                        f"YAML names will be used for metrics table. Visualizations will use names from '{labels_txt_path}' (via ObjectDetectionUtils)."
                    )
            elif class_names_from_txt_file:
                authoritative_class_names = class_names_from_txt_file
                logger.warning(
                    f"data.yaml ('{data_yaml_path}') does not contain a 'names' list. "
                    f"Falling back to class names from --labels file ('{labels_txt_path}') as authoritative for metrics. "
                    "Ensure this is consistent with your model and ground truth IDs."
                )
            else: 
                logger.error(f"CRITICAL: No class names available from data.yaml or '{labels_txt_path}'. Cannot proceed with validation accurately.")
                run_validation_metrics = False 
        else:
            logger.warning(f"Failed to load images for validation from '{data_yaml_path}'. Disabling metrics mode.")
            run_validation_metrics = False
    
    if not run_validation_metrics:
        logger.info(f"Normal inference mode. Using input: {input_source}")
        if not authoritative_class_names: 
             authoritative_class_names = class_names_from_txt_file
        
        if input_source.lower() == "camera":
            temp_cap = cv2.VideoCapture(0)
            if not temp_cap.isOpened():
                logger.error("Cannot open camera")
                if temp_cap: temp_cap.release()
                return
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)
            cap = temp_cap
            image_file_paths_for_inference = [] 
        elif Path(input_source).is_file() and Path(input_source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            temp_cap = cv2.VideoCapture(input_source)
            if not temp_cap.isOpened():
                logger.error(f"Cannot open video file: {input_source}")
                if temp_cap: temp_cap.release()
                return
            cap = temp_cap
            image_file_paths_for_inference = []
        else: 
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

    if not authoritative_class_names:
        logger.error("CRITICAL: Authoritative class names could not be established. Exiting.")
        return

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

    if run_validation_metrics and authoritative_class_names:
        if all_hailo_predictions and current_run_ground_truth_map: 
            logger.info("Calculating validation metrics...")
            # Call the imported function
            calculate_and_print_metrics_table(
                all_hailo_predictions, 
                current_run_ground_truth_map, 
                authoritative_class_names
            )
        else:
            logger.warning("Not enough data for validation metrics (predictions missing or ground truth map empty).")

    logger.info('Inference/Validation run completed.')


def main() -> None:
    args = parse_args()
    infer(
        input_source=args.input,
        save_stream_output=args.save_stream_output,
        net_path=args.net,
        labels_txt_path=args.labels, # Pass the path to the .txt labels file
        batch_size=args.batch_size,
        data_yaml_path=args.data,
        use_test_split=args.test_split 
    )

if __name__ == "__main__":
    main()