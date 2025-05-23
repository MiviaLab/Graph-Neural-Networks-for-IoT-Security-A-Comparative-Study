#!/bin/bash

cd ../model
#### ---- tdg 150 ---- ####
MODEL=[EXPERIMENT_NAME] # DOMINANT_B4_64_NORM_150k_IoT23_tdg
BASE_PATH=[PATH_TO_CHECKPOINTS_DIR]
NORMALIZE=1

CHECKPOINT_FOLDER=${BASE_PATH}/${MODEL}/checkpoints
CSV_RESULT_PATH=${BASE_PATH}/${MODEL}/csv_results
RESULT_PATH=${BASE_PATH}/${MODEL}/y_pred_true
THRESHOLD_PATH=${BASE_PATH}/${MODEL}/thresholds

MIN_MAX=[PATH_TO_MIN_MAX_TRAIN] #"[PATH_TO_anomaly_detection_dataset]/min_max_benign/150k/IoT23_min_max_benign"
GRAPH_TYPE="tdg_graph"
DATASET_FOLDER=[PATH_TO_DATASET] #"[PATH_TO_anomaly_detection_dataset]/150000/IoT23/base"
DATASET="IoT23"
JSON_FOLDER=[PATH_TO_TRAIN_VAL_TEST_SPLIT] #"[PATH_TO_anomaly_detection_dataset]/split/150k/base/IoT23_dataset_split_etdg"
python test.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_type ${GRAPH_TYPE} --checkpoint_path ${CHECKPOINT_FOLDER} --result_path ${RESULT_PATH} --min_max ${MIN_MAX} --dataset ${DATASET} --normalize ${NORMALIZE} --threshold_path ${THRESHOLD_PATH} --compute_evaluation_metrics_flag

DATASET_FOLDER=[PATH_TO_DATASET] #"[PATH_TO_anomaly_detection_dataset]/150000/IoT_traces/base"
DATASET="IoT_traces"
JSON_FOLDER=[PATH_TO_TRAIN_VAL_TEST_SPLIT] #"[PATH_TO_anomaly_detection_dataset]/split_test/150k/base/IoT_traces_dataset_split_etdg"
python test.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_type ${GRAPH_TYPE} --checkpoint_path ${CHECKPOINT_FOLDER} --result_path ${RESULT_PATH} --min_max ${MIN_MAX} --dataset ${DATASET} --normalize ${NORMALIZE} --threshold_path ${THRESHOLD_PATH}

DATASET_FOLDER=[PATH_TO_DATASET] #"[PATH_TO_anomaly_detection_dataset]/150000/IoTID20/base"
DATASET="IoTID20"
JSON_FOLDER=[PATH_TO_TRAIN_VAL_TEST_SPLIT] #"[PATH_TO_anomaly_detection_dataset]/split_test/150k/base/IoTID20_split_etdg"
python test.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_type ${GRAPH_TYPE} --checkpoint_path ${CHECKPOINT_FOLDER} --result_path ${RESULT_PATH} --min_max ${MIN_MAX} --dataset ${DATASET} --normalize ${NORMALIZE} --threshold_path ${THRESHOLD_PATH}
