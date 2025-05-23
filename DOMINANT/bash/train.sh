#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

cd ../model

MODEL=[EXPERIMENT_NAME] # DOMINANT_B4_64_NORM_150k_IoT23_etdg
NORMALIZE=1
DATASET_FOLDER=[PATH_TO_DATASET]             #"[PATH_TO_anomaly_detection_dataset]/150000/IoT23/base"
JSON_FOLDER=[PATH_TO_TRAIN_VAL_TEST_SPLIT]   #"[PATH_TO_anomaly_detection_dataset]/split/150k/base/IoT23_dataset_split_etdg"
MIN_MAX=[PATH_TO_MIN_MAX_TRAIN]              #"[PATH_TO_anomaly_detection_dataset]/min_max_benign/150k/IoT23_min_max_benign"
CHECKPOINT_FOLDER=[PATH_TO_SAVE_CHECKPOINTS] #"[PATH_TO_checkpoints_dir]/${MODEL}/checkpoints"
CSV_RESULT_PATH=[PATH_TO_SAVE_CSV_LOGS]      #"[PATH_TO_checkpoints_dir]/${MODEL}/checkpoints"

GRAPH_TYPE=["etdg_graph" or "tdg_graph"]
python train.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_typ ${GRAPH_TYPE} --checkpoint_folder ${CHECKPOINT_FOLDER} --csv_results_folder ${CSV_RESULT_PATH} --normalize ${NORMALIZE} --model ${MODEL} --min_max ${MIN_MAX}
