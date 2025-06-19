#!/usr/bin/env python3

# offline_evaluator.py
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import sys
import os

from loguru import logger # Assuming loguru is your preferred logger

# Import necessary functions from your existing validation module
# Ensure object_detection_val.py is in the same directory or Python path
try:
    from object_detection_val import (
        load_ground_truth_data,
        calculate_and_print_metrics_table_acc_numba_mproc,
        calculate_and_print_metrics_table_acc_pyloop,
        log_collection_memory_usage
    )
except ImportError:
    logger.error("Failed to import from object_detection_val.py. "
                 "Ensure the file is in the correct path, its dependencies are installed, "
                 "and it can find 'utils.py' (sys.path should be set by now).")
    logger.exception("Detailed import error for object_detection_val:") # Provides full traceback
    exit(1)
except Exception as e: # Catch other potential errors during import phase
    logger.error(f"An unexpected error occurred while importing object_detection_val: {e}")
    logger.exception("Detailed error:")
    exit(1)


def parse_offline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Metrics Evaluator for Hailo Detections")
    parser.add_argument(
        "--metrics-version",
        choices=["pyloop", "numba_mproc"],
        default="numba_mproc",
        help="Which metrics implementation to use: 'pyloop' (original Python loop) or 'numba_mproc' (Numba + multiprocessing, default)."
    )    
    parser.add_argument(
        "--data_yaml",
        required=True,
        type=str,
        help="Path to the data.yaml file (Ultralytics format) for ground truth and class names."
    )
    parser.add_argument(
        "--predictions_json",
        required=True,
        type=str,
        help="Path to the JSON file containing the exported Hailo predictions."
    )
    parser.add_argument(
        "--test_split",
        action="store_true",
        help="Use the 'test' split from data.yaml for ground truth instead of 'val'."
    )
    # Add any other arguments you might need, e.g., specific output directory for metrics results
    return parser.parse_args()


def load_predictions_from_json(filepath: str) -> Optional[List[Dict[str, Any]]]:
    """Loads predictions from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            predictions = json.load(f)
        logger.info(f"Successfully loaded predictions from: {filepath}")
        return predictions
    except FileNotFoundError:
        logger.error(f"Predictions JSON file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file \'{filepath}\': {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load predictions from JSON file \'{filepath}\': {e}")
        return None


def main_offline_eval():
    args = parse_offline_args()

    logger.info(f"Starting offline evaluation...")
    logger.info(f"Data YAML: {args.data_yaml}")
    logger.info(f"Predictions JSON: {args.predictions_json}")
    logger.info(f"Using test split: {args.test_split}")

    # Determine dataset root for this offline evaluation context (current yaml file)
    offline_dataset_root: Optional[Path] = None
    try:
        with open(args.data_yaml, 'r') as f_yaml_offline_root:
            yaml_cfg_offline = yaml.safe_load(f_yaml_offline_root)
        if yaml_cfg_offline and 'path' in yaml_cfg_offline:
            path_from_yaml = Path(yaml_cfg_offline['path'])
            if path_from_yaml.is_absolute():
                offline_dataset_root = path_from_yaml.resolve()
            else:
                offline_dataset_root = (Path(args.data_yaml).parent / path_from_yaml).resolve()
            logger.info(f"Offline evaluator determined dataset root: {offline_dataset_root}")
        else:
            logger.error(f"'path' key not found in the data YAML '{args.data_yaml}'. Cannot reconstruct absolute image paths.")
            return
    except Exception as e:
        logger.error(f"Error reading dataset root 'path' from '{args.data_yaml}': {e}")
        return

    # 1. Load ground truth data
    logger.info("Loading ground truth data...")
    # load_ground_truth_data returns: Tuple[List[str], Dict[str, Dict[str, Any]]]
    # (image_file_paths, ground_truth_map)
    # We primarily need the ground_truth_map for metrics.
    _image_paths_from_gt, ground_truth_map = load_ground_truth_data(args.data_yaml, args.test_split)

    if not ground_truth_map:
        logger.error("Failed to load ground truth data or ground truth map is empty. Exiting.")
        return

    # Log memory of ground truth map
    log_collection_memory_usage("ground_truth_map (offline evaluator)", ground_truth_map, logger)

    # 2. Load authoritative class names from the YAML
    # This logic should ideally be robust within load_ground_truth_data or a shared utility,
    # but for now, we'll parse it here as calculate_and_print_metrics_table expects class_names.
    authoritative_class_names: Optional[List[str]] = None
    try:
        with open(args.data_yaml, 'r') as f_yaml:
            yaml_cfg = yaml.safe_load(f_yaml) # Assuming pyyaml is installed
        if yaml_cfg and 'names' in yaml_cfg:
            names_field = yaml_cfg['names']
            if isinstance(names_field, dict):
                if all(isinstance(k, int) for k in names_field.keys()):
                    sorted_keys = sorted(names_field.keys())
                    if sorted_keys == list(range(len(sorted_keys))): # Check for 0-indexed contiguous
                        authoritative_class_names = [names_field[k] for k in sorted_keys]
                    else: # Handle non-contiguous or non-0-indexed if necessary, or warn
                        logger.warning(f"YAML 'names' dictionary keys are not 0-indexed contiguous. Using sorted key order.")
                        authoritative_class_names = [names_field[k] for k in sorted_keys] # May not align if IDs are arbitrary
                else:
                    logger.warning("YAML 'names' is a dictionary, but keys are not all integers.")
            elif isinstance(names_field, list):
                authoritative_class_names = names_field
        
        if not authoritative_class_names and yaml_cfg and 'nc' in yaml_cfg: # Fallback to nc for placeholder names
            logger.warning(f"Could not parse 'names' from YAML. Using 'nc' ({yaml_cfg['nc']}) to generate placeholder names if labels.txt is not used by metrics func.")
            # Note: calculate_and_print_metrics_table requires actual names.
            # This fallback path for authoritative_class_names should be handled carefully.
            # For simplicity, we assume 'names' will be present as per your earlier script logic.

        if not authoritative_class_names:
            logger.error(f"Could not determine authoritative class names from {args.data_yaml}. Metrics table will be affected.")
            # Potentially try to load from a default labels.txt if that's a fallback you want.
            # For now, we rely on the YAML.
            return

    except Exception as e:
        logger.error(f"Error reading or parsing class names from data_yaml '{args.data_yaml}': {e}")
        return

    # 3. Load exported Hailo predictions
    logger.info("Loading Hailo predictions...")
    raw_hailo_predictions = load_predictions_from_json(args.predictions_json)

    if not raw_hailo_predictions:
        logger.error("Failed to load Hailo predictions. Exiting.")
        return

    log_collection_memory_usage("hailo_predictions (offline evaluator)", raw_hailo_predictions, logger)

    # Reconstruct absolute paths for predictions ---
    reconstructed_hailo_predictions: List[Dict[str, Any]] = []
    if offline_dataset_root:
        for pred_entry in raw_hailo_predictions:
            path_from_json = pred_entry['image_path']
            # Path() can usually handle POSIX paths fine on Windows and vice-versa for joining
            # but resolve() makes it canonical for the current OS.
            try:
                # Check if path_from_json is already absolute (fallback case from object_detection.py)
                if Path(path_from_json).is_absolute():
                    abs_image_path = Path(path_from_json).resolve()
                else:
                    abs_image_path = (offline_dataset_root / path_from_json).resolve()
                
                # Create a new entry to avoid modifying the original list if it's used elsewhere
                updated_pred_entry = pred_entry.copy()
                updated_pred_entry['image_path'] = str(abs_image_path) # Store as string
                reconstructed_hailo_predictions.append(updated_pred_entry)
            except Exception as e:
                logger.error(f"Error reconstructing path for '{path_from_json}' with root '{offline_dataset_root}': {e}. Skipping entry.")
        
        logger.info(f"Reconstructed paths for {len(reconstructed_hailo_predictions)} prediction entries.")
        if reconstructed_hailo_predictions: # Log a sample reconstructed path
            logger.debug(f"Sample reconstructed prediction image_path: {reconstructed_hailo_predictions[0]['image_path']}")

    else: # Should not happen if YAML parsing for root was successful
        logger.error("Offline dataset root not determined. Cannot reconstruct prediction paths.")
        return
        
    log_collection_memory_usage("hailo_predictions (reconstructed paths)", reconstructed_hailo_predictions, logger)

    # 4. Calculate and print metrics
    if reconstructed_hailo_predictions and ground_truth_map and authoritative_class_names:
        logger.info("Calculating validation metrics...")
        # The choice of metrics implementation is made based on the command line argument.
        # More metrics implementations can be added by extending the if-else logic here.
        # This allows for flexibility in testing and performance evaluation.
        if args.metrics_version == "numba_mproc":
            logger.info("Metrics calculated using Numba + multiprocessing.")
            calculate_and_print_metrics_table_acc_numba_mproc(
                reconstructed_hailo_predictions,
                ground_truth_map,
                authoritative_class_names
            )
        elif args.metrics_version == "pyloop":
            # This is the original Python loop implementation for metrics calculation
            # It is faster for very dense classes
            logger.info("Metrics calculated using original Python loop (pyloop).")
            calculate_and_print_metrics_table_acc_pyloop(
                reconstructed_hailo_predictions,
                ground_truth_map,
                authoritative_class_names
            )
    else:
        logger.warning("Not enough data to calculate offline validation metrics (predictions, ground truth, or class names missing).")

    logger.info("Offline evaluation completed.")


if __name__ == "__main__":
    # Configure logger (optional, but good practice)
    # You might have a shared logging setup
    logger.remove() # Removes default handler
    logger.add(sys.stderr, level="INFO") # Add a basic handler
    
    main_offline_eval()
