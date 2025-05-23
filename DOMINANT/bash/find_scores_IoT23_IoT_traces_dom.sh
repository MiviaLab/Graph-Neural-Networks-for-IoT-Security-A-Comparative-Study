#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

configurations=(
    

    ## FEATURE IMPORTANCE 
    #"feature_importance_dataset/150k/only_etdg/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_only_etdg" "1" "150k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/150k/only_etdg/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/150k/top23/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_top_23" "1" "150k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/150k/top23/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/150k/top53/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_top_53" "1" "150k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/150k/top53/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/150k/top63/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_top_63" "1" "150k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/150k/top63/IoT23_min_max_benign" "etdg_graph"

    "feature_importance_dataset/120k/only_etdg/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_only_etdg" "1" "120k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/120k/only_etdg/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/120k/top23/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_top_23" "1" "120k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/120k/top23/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/120k/top53/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_top_53" "1" "120k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/120k/top53/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/120k/top63/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_top_63" "1" "120k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/120k/top63/IoT23_min_max_benign" "etdg_graph"
    
    #"feature_importance_dataset/90k/only_etdg/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_only_etdg" "1" "90k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/90k/only_etdg/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/90k/top23/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_top_23" "1" "90k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/90k/top23/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/90k/top53/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_top_53" "1" "90k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/90k/top53/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/90k/top63/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_top_63" "1" "90k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/90k/top63/IoT23_min_max_benign" "etdg_graph"

    "feature_importance_dataset/60k/only_etdg/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_only_etdg" "1" "60k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/60k/only_etdg/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/60k/top23/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_top_23" "1" "60k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/60k/top23/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/60k/top53/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_top_53" "1" "60k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/60k/top53/IoT23_min_max_benign" "etdg_graph"
    #"feature_importance_dataset/60k/top63/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_top_63" "1" "60k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/60k/top63/IoT23_min_max_benign" "etdg_graph"
    
    #"150000/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_etdg" "1" "150k/base/IoT23_dataset_split_etdg" "150k/IoT23_min_max_benign" "etdg_graph"
    #"120000/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_etdg" "1" "120k/base/IoT23_dataset_split_etdg" "120k/IoT23_min_max_benign" "etdg_graph"
    #"90000/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_etdg" "1" "90k/base/IoT23_dataset_split_etdg" "90k/IoT23_min_max_benign" "etdg_graph"
    #"60000/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_etdg" "1" "60k/base/IoT23_dataset_split_etdg" "60k/IoT23_min_max_benign" "etdg_graph"

    #"150000/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_tdg" "1" "150k/base/IoT23_dataset_split_etdg" "150k/IoT23_min_max_benign" "tdg_graph"
    #"120000/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_tdg" "1" "120k/base/IoT23_dataset_split_etdg" "120k/IoT23_min_max_benign" "tdg_graph"
    #"90000/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_tdg" "1" "90k/base/IoT23_dataset_split_etdg" "90k/IoT23_min_max_benign" "tdg_graph"
    #"60000/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_tdg" "1" "60k/base/IoT23_dataset_split_etdg" "60k/IoT23_min_max_benign" "tdg_graph"
    #"10000/IoT23/base" "DOMINANT_B4_64_NORM_10k_IoT23_tdg" "1" "10k/base/IoT23_dataset_split_tdg" "10k/IoT23_min_max_benign" "tdg_graph"

    #"150000/IoT23/base" "DOMINANT_B4_64_NO_NORM_150k_IoT23_tdg" "0" "150k/base/IoT23_dataset_split_tdg" "150k/IoT23_min_max_benign" "tdg_graph"
    #"120000/IoT23/base" "DOMINANT_B4_64_NO_NORM_120k_IoT23_tdg" "0" "120k/base/IoT23_dataset_split_tdg" "120k/IoT23_min_max_benign" "tdg_graph"
    #"90000/IoT23/base" "DOMINANT_B4_64_NO_NORM_90k_IoT23_tdg" "0" "90k/base/IoT23_dataset_split_tdg" "90k/IoT23_min_max_benign" "tdg_graph"
    #"60000/IoT23/base" "DOMINANT_B4_64_NO_NORM_60k_IoT23_tdg" "0" "60k/base/IoT23_dataset_split_tdg" "60k/IoT23_min_max_benign" "tdg_graph"
    #"10000/IoT23/base" "DOMINANT_B4_64_NO_NORM_10k_IoT23_tdg" "0" "10k/base/IoT23_dataset_split_tdg" "10k/IoT23_min_max_benign" "tdg_graph"

    #"150000/IoT_traces/base" "DOMINANT_B4_64_NORM_150k_IoT_traces_etdg" "1" "150k/base/IoT_traces_dataset_split_etdg" "150k/IoT_traces_min_max_benign" "etdg_graph"
    #"120000/IoT_traces/base" "DOMINANT_B4_64_NORM_120k_IoT_traces_etdg" "1" "120k/base/IoT_traces_dataset_split_etdg" "120k/IoT_traces_min_max_benign" "etdg_graph"
    #"90000/IoT_traces/base" "DOMINANT_B4_64_NORM_90k_IoT_traces_etdg" "1" "90k/base/IoT_traces_dataset_split_etdg" "90k/IoT_traces_min_max_benign" "etdg_graph"
    #"60000/IoT_traces/base" "DOMINANT_B4_64_NORM_60k_IoT_traces_etdg" "1" "60k/base/IoT_traces_dataset_split_etdg" "60k/IoT_traces_min_max_benign" "etdg_graph"

    #"150000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_150k_IoT_traces_etdg" "0" "150k/base/IoT_traces_dataset_split_etdg" "150k/IoT_traces_min_max_benign" "etdg_graph"
    #"120000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_120k_IoT_traces_etdg" "0" "120k/base/IoT_traces_dataset_split_etdg" "120k/IoT_traces_min_max_benign" "etdg_graph"
    #"90000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_90k_IoT_traces_etdg" "0" "90k/base/IoT_traces_dataset_split_etdg" "90k/IoT_traces_min_max_benign" "etdg_graph"
    #"60000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_60k_IoT_traces_etdg" "0" "60k/base/IoT_traces_dataset_split_etdg" "60k/IoT_traces_min_max_benign" "etdg_graph"

    #"150000/IoT_traces/base" "DOMINANT_B4_64_NORM_150k_IoT_traces_tdg" "1" "150k/base/IoT_traces_dataset_split_tdg" "150k/IoT_traces_min_max_benign" "tdg_graph"
    #"120000/IoT_traces/base" "DOMINANT_B4_64_NORM_120k_IoT_traces_tdg" "1" "120k/base/IoT_traces_dataset_split_tdg" "120k/IoT_traces_min_max_benign" "tdg_graph"
    #"90000/IoT_traces/base" "DOMINANT_B4_64_NORM_90k_IoT_traces_tdg" "1" "90k/base/IoT_traces_dataset_split_tdg" "90k/IoT_traces_min_max_benign" "tdg_graph"
    #"60000/IoT_traces/base" "DOMINANT_B4_64_NORM_60k_IoT_traces_tdg" "1" "60k/base/IoT_traces_dataset_split_tdg" "60k/IoT_traces_min_max_benign" "tdg_graph"
    #"10000/IoT_traces/base" "DOMINANT_B4_64_NORM_10k_IoT_traces_tdg" "1" "10k/base/IoT_traces_dataset_split_tdg" "10k/IoT_traces_min_max_benign" "tdg_graph"

    #"150000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_150k_IoT_traces_tdg" "0" "150k/base/IoT_traces_dataset_split_tdg" "150k/IoT_traces_min_max_benign" "tdg_graph"
    #"120000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_120k_IoT_traces_tdg" "0" "120k/base/IoT_traces_dataset_split_tdg" "120k/IoT_traces_min_max_benign" "tdg_graph"
    #"90000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_90k_IoT_traces_tdg" "0" "90k/base/IoT_traces_dataset_split_tdg" "90k/IoT_traces_min_max_benign" "tdg_graph"
    #"60000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_60k_IoT_traces_tdg" "0" "60k/base/IoT_traces_dataset_split_tdg" "60k/IoT_traces_min_max_benign" "tdg_graph"
    #"10000/IoT_traces/base" "DOMINANT_B4_64_NO_NORM_10k_IoT_traces_tdg" "0" "10k/base/IoT_traces_dataset_split_tdg" "10k/IoT_traces_min_max_benign" "tdg_graph"

    #"150000/IoT23/base" "DOMINANT_B4_64_NO_NORM_150k_IoT23_etdg" "0" "150k/base/IoT23_dataset_split_etdg" "150k/IoT23_min_max_benign" "etdg_graph"
    #"120000/IoT23/base" "DOMINANT_B4_64_NO_NORM_120k_IoT23_etdg" "0" "120k/base/IoT23_dataset_split_etdg" "120k/IoT23_min_max_benign" "etdg_graph"
    #"90000/IoT23/base" "DOMINANT_B4_64_NO_NORM_90k_IoT23_etdg" "0" "90k/base/IoT23_dataset_split_etdg" "90k/IoT23_min_max_benign" "etdg_graph"
    #"60000/IoT23/base" "DOMINANT_B4_64_NO_NORM_60k_IoT23_etdg" "0" "60k/base/IoT23_dataset_split_etdg" "60k/IoT23_min_max_benign" "etdg_graph"

)

cd ../model
# Iterate over configurations

for ((i=0; i<${#configurations[@]}; i+=6)); do
    DATA_FOLD="${configurations[i]}"
    MODEL="${configurations[i+1]}"
    NORMALIZE="${configurations[i+2]}"
    JSON_FOLD="${configurations[i+3]}"
    MM="${configurations[i+4]}"
    GRAPH_TYPE="${configurations[i+5]}"
    
    DATASET_FOLDER="/user/apaolillo/Output_Grafi/$DATA_FOLD"
    JSON_FOLDER="/user/apaolillo/Output_Grafi/split/$JSON_FOLD"
    MIN_MAX="/user/apaolillo/Output_Grafi/$MM"
    #MIN_MAX="/user/apaolillo/Output_Grafi/min_max_benign/$MM"
    CHECKPOINT_FOLDER="/user/apaolillo/checkpoint_dir/dominant/$MODEL/checkpoints"
    THRESHOLD_PATH="/user/apaolillo/checkpoint_dir/dominant/$MODEL/thresholds"
    echo "Work on: $MODEL"
    python find_scores.py --dataset_folder "${DATASET_FOLDER}" --json_folder "${JSON_FOLDER}" --graph_type "${GRAPH_TYPE}" --checkpoint_path "${CHECKPOINT_FOLDER}"  --min_max "${MIN_MAX}" --normalize "${NORMALIZE}" --threshold_path "${THRESHOLD_PATH}" 
done
