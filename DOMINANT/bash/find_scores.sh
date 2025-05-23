#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

cd ../model
#MODEL=DOMINANT_B4_64_NO_NORM
#NORMALIZE=0
#MIN_MAX=/user/frosa/anomaly_detection_code/gnn-network-analysis/IDS/datasets/IoT23_min_max
#CHECKPOINT_FOLDER=/user/frosa/anomaly_detection_code/checkpoint_dir/${MODEL}/checkpoints
#CSV_RESULT_PATH=/user/frosa/anomaly_detection_code/checkpoint_dir/${MODEL}/csv_results
#RESULT_PATH=/user/frosa/anomaly_detection_code/checkpoint_dir/${MODEL}/y_pred_true
#THRESHOLD_PATH=/user/frosa/anomaly_detection_code/checkpoint_dir/${MODEL}/thresholds

# ---- ETDG ----
#GRAPH_TYPE="tdg_graph"
#DATASET_FOLDER="/user/frosa/anomaly_dataset/IoT23/dataset/IoT23_graphs/"
#JSON_FOLDER="/user/frosa/anomaly_detection_code/gnn-network-analysis/IDS/datasets/IoT23_dataset_split"
#python find_scores.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_type ${GRAPH_TYPE} --checkpoint_path ${CHECKPOINT_FOLDER} --min_max ${MIN_MAX} --normalize ${NORMALIZE} --threshold_path ${THRESHOLD_PATH} 

#GRAPH_TYPE="etdg_graph"
#DATASET_FOLDER="/user/frosa/anomaly_dataset/IoT23/dataset/IoT23_graphs/"
#JSON_FOLDER="/user/frosa/anomaly_detection_code/gnn-network-analysis/IDS/datasets/IoT23_dataset_split"
#python find_scores.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_type ${GRAPH_TYPE} --checkpoint_path ${CHECKPOINT_FOLDER} --min_max ${MIN_MAX} --normalize ${NORMALIZE} --threshold_path ${THRESHOLD_PATH} 

# ------------------------------------- #
MODEL=DOMINANT_B4_64_NORM_150k_IoT23_etdg
NORMALIZE=1
MIN_MAX=/user/apaolillo/Output_Grafi/min_max_benign/150k/IoT23_min_max_benign
CHECKPOINT_FOLDER=/user/apaolillo/checkpoint_dir/${MODEL}/checkpoints
CSV_RESULT_PATH=/user/apaolillo/checkpoint_dir/${MODEL}/csv_results
RESULT_PATH=/user/apaolillo/checkpoint_dir/${MODEL}/y_pred_true
THRESHOLD_PATH=/user/apaolillo/checkpoint_dir/${MODEL}/thresholds

# ---- ETDG ----
#GRAPH_TYPE="tdg_graph"
#DATASET_FOLDER="/user/frosa/anomaly_dataset/IoT23/dataset/IoT23_graphs/"
#JSON_FOLDER="/user/frosa/anomaly_detection_code/gnn-network-analysis/IDS/datasets/IoT23_dataset_split"
#python find_scores.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_type ${GRAPH_TYPE} --checkpoint_path ${CHECKPOINT_FOLDER} --min_max ${MIN_MAX} --normalize ${NORMALIZE} --threshold_path ${THRESHOLD_PATH} 

GRAPH_TYPE="etdg_graph"
DATASET_FOLDER="/user/apaolillo/Output_Grafi/150000/IoT23/base/"
JSON_FOLDER="/user/apaolillo/Output_Grafi/split/150k/base/IoT23_dataset_split_etdg"
python find_scores.py --dataset_folder ${DATASET_FOLDER} --json_folder ${JSON_FOLDER} --graph_type ${GRAPH_TYPE} --checkpoint_path ${CHECKPOINT_FOLDER} --min_max ${MIN_MAX} --normalize ${NORMALIZE} --threshold_path ${THRESHOLD_PATH} 

cd ../bash
#nohup ./test_norm.sh > test_norm.txt &
#nohup ./test.sh > test_no_norm.txt &