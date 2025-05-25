#!/bin/bash

BASE_PATH=[BASE_PATH_TO_PROJECT_DIR]

cd ../model

SNAP=[SNAPSHOT_SIZE] # e.g., 150, 120, 90, 60
DATA_FOLD="${SNAP}000/IoT23/base"
MODEL=[EXPERIMENT_NAME] #"OCGNN_B4_64_NORM_${SNAP}k_IoT23_etdg_debug_new_update_radius_policy_save_best_model_val_accuracy"
NORMALIZE="1"
JSON_FOLD="${SNAP}k/base/IoT23_dataset_split_etdg"
MM="min_max_benign/${SNAP}k/IoT23_min_max_benign"
GRAPH_TYPE="etdg_graph" # ["etdg_graph", "tdg_graph"]

DATASET_FOLDER="${BASE_PATH}/anomaly_detection_dataset/$DATA_FOLD"
JSON_FOLDER="${BASE_PATH}/anomaly_detection_dataset/split/$JSON_FOLD"
MIN_MAX="${BASE_PATH}/anomaly_detection_dataset/$MM"
#MIN_MAX="${BASE_PATH}/min_max_benign/$MM"
CHECKPOINT_FOLDER="${BASE_PATH}/anomaly_detection_code_checkpoint/Checkpoint_dataset_snap/ocgcn/$MODEL/checkpoints"
CSV_RESULT_PATH="${BASE_PATH}/anomaly_detection_code_checkpoint/Checkpoint_dataset_snap/ocgcn/$MODEL/csv_results"
CENTER_PATH="${BASE_PATH}/anomaly_detection_code_checkpoint/Checkpoint_dataset_snap/ocgcn/$MODEL/center"
RADIUS_PATH="${BASE_PATH}/anomaly_detection_code_checkpoint/Checkpoint_dataset_snap/ocgcn/$MODEL/radius"

python train.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_typ ${GRAPH_TYPE} --checkpoint_folder ${CHECKPOINT_FOLDER} --csv_results_folder ${CSV_RESULT_PATH} --center_path ${CENTER_PATH} --radius_path ${RADIUS_PATH} --normalize ${NORMALIZE} --min_max ${MIN_MAX} --model ${MODEL} --wandb_log
