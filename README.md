# Documentation of post-quantization validation code

## Overview

This a usage guide for the yolo/voc-style validation of object detection models quantized and compiled for Hailo-8/8L accelerators. The code is in the `obj-det-val` branch of a fork of the `Hailo-Application-Code-Examples` and embedded in the `object-detection` example. There are two stages to the process, each having its own Python script, Python environment requirements, and executiong platform:
1. Gathering of inference output (detections) on an RPi 5 connected to a Hailo accelerator, and export in a JSON format.
2. Computation of YOLO/VOC-style validation metrics on a regular PC or server, using the exported JSON file.

## Notable files

1. [`object_detection.py`](runtime/hailo-8/python/object_detection/object_detection.py) is the script which runs inference on the Hailo accelerator with a compiled (`.hef`) model against a dataset, accumulates the detections and exports them to a JSON file with a distinguishable long name.
2. [`object_detection_val.py`](runtime/hailo-8/python/object_detection/object_detection_val.py) contains all of the validation code added to the directory. This is all new code. It is used in both stages.
3. [`object_detection_utils.py`](runtime/hailo-8/python/object_detection/object_detection_utils.py) contains object detection ultility functions. Almost unchanged.
4. [`../utils.py`](runtime/hailo-8/python/utils.py) contains the inference code using the Hailo platform in the form of the Python API of HailoRT.
5. [`offline_evaluator.py`](runtime/hailo-8/python/object_detection/offline_evaluator.py) contains the code that performs the validation based on the dataset ground truth and the predictions (detection) contained in the JSON file.

## Stage 1: Inference output and JSON export

1. Runs [`object_detection.py`](runtime/hailo-8/python/object_detection/object_detection.py) on the RPi 5 with the Hailo accelerator and the Hailo stack installed. It runs inference for a model and a dataset, using a `txt` file of class labels in the format
   ```
   class0
   class1
   class2
   ...
   ```
2. Optionally collects the detections and compiles a JSON file for export for offline validation metrics calculation on a faster machine. In this case it takes a YOLO-style `yaml` file of object classes passed with the `--data` argument, in the format
   ```
   path: path-to-dataset-to-use-in-inference  # best to be an absolute path
   train: images/train  # relative to path
   val: images/val
   test: images/test
   
   names:
     0: class0
     1: class1
     2: class2
     ...
   ```
3. It can output annotated images. It can take a single image, a directory of images, a video, or a camera stream passed with the `-i` argument. This is ignored when `--data` is passed. If there is discrepancy between the `labels.txt` file and the `classes.yaml` file, the latter holds precedence for detection collection and export.
4. This stage requires the `hailo-platform` (see the [`../utils.py`](runtime/hailo-8/python/utils.py) file). Therefore set up the environment at the top level of the repository as follows
   ```
   cd Hailo-Application-Code-Examples
   git switch obj-det-val
   source setup_env.sh
   ```
5. To run the object detection with detection collection and JSON export, download your model and dataset (formatted YOLO-style), reference the latter correctly in the `classes.yaml` file, and
   ```
   cd runtime/hailo-8/python/object_detection
   python object_detection.py --help
   python object_detection.py -n path/to/model/hef -l path/to/labels/txt --data path/to/dataset/yaml
   ```
6. The detections are written to a large JSON file in the directory `predictions_export`. This can be transfered to a more powerful machine to run the next stage of the validation, the actual calculation of the YOLO/VOC-style metrics.
7. By default, the script will reference and use the `val` split of the dataset. Optionally, the `test` split can be used instead by passing the boolean argument `--test_split`.

## Stage 2: Calculation of validation metrics and printing table

1. Runs [`offline_evaluator.py`](runtime/hailo-8/python/object_detection/offline_evaluator.py) on a PC, where the JSON file and dataset from Stage 1 are downloaded and the latter set up in the `classes.yaml` file.
2. To set up the environment
   ```
   cd Hailo-Application-Code-Examples
   python -m venv .offline_val_venv
   source .offline_val_venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements-offline-val.txt
   ```
3. To run
   ```
   cd runtime/hailo
   cd runtime/hailo-8/python/object_detection
   python offline_evaluator.py --help
   python offline_evaluator.py --metrics-version pyloop --predictions-json path/to/downloaded/predictions/json --data-yaml path/to/the/dataset/and/classes/yaml
   ```
4. If the JSON file contains test split predictions, specify this with the boolean argument `--test_split`.
5. There are two separate implementations of the metrics calculation, `numba_mproc` and `pyloop`. The latter is slightly faster.
6. The end of the output contains a YOLO-style validation metrics table.
 
