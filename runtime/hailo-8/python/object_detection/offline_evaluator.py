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
        calculate_and_print_metrics_table,
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
    hailo_predictions = load_predictions_from_json(args.predictions_json)

    if not hailo_predictions:
        logger.error("Failed to load Hailo predictions. Exiting.")
        return

    log_collection_memory_usage("hailo_predictions (offline evaluator)", hailo_predictions, logger)


    # 4. Calculate and print metrics
    if hailo_predictions and ground_truth_map and authoritative_class_names:
        logger.info("Calculating validation metrics...")
        calculate_and_print_metrics_table(
            hailo_predictions,
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
