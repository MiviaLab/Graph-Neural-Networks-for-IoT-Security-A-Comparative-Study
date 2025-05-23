#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

configurations=(
    ## FEATURE IMPORTANCE 
    #"feature_importance_dataset/150k/only_etdg/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_only_etdg" "1" "150k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/150k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/150k/only_etdg/IoTID20/base" "DOMINANT_B4_64_NORM_150k_IoT23_only_etdg" "1" "150k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/150k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/150k/only_etdg/IoT_traces/base" "DOMINANT_B4_64_NORM_150k_IoT23_only_etdg" "1" "150k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/150k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/150k/top23/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_top23" "1" "150k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/150k/top23/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/150k/top23/IoTID20/base" "DOMINANT_B4_64_NORM_150k_IoT23_top23" "1" "150k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/150k/top23/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/150k/top23/IoT_traces/base" "DOMINANT_B4_64_NORM_150k_IoT23_top23" "1" "150k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/150k/top23/IoT23_min_max_benign" "etdg_graph" "IoT_traces"
   
    #"feature_importance_dataset/150k/top53/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_top53" "1" "150k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/150k/top53/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/150k/top53/IoTID20/base" "DOMINANT_B4_64_NORM_150k_IoT23_top53" "1" "150k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/150k/top53/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/150k/top53/IoT_traces/base" "DOMINANT_B4_64_NORM_150k_IoT23_top53" "1" "150k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/150k/top53/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/150k/top63/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_top63" "1" "150k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/150k/top63/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/150k/top63/IoTID20/base" "DOMINANT_B4_64_NORM_150k_IoT23_top63" "1" "150k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/150k/top63/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/150k/top63/IoT_traces/base" "DOMINANT_B4_64_NORM_150k_IoT23_top63" "1" "150k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/150k/top63/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/120k/only_etdg/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_only_etdg" "1" "120k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/120k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/120k/only_etdg/IoTID20/base" "DOMINANT_B4_64_NORM_120k_IoT23_only_etdg" "1" "120k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/120k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/120k/only_etdg/IoT_traces/base" "DOMINANT_B4_64_NORM_120k_IoT23_only_etdg" "1" "120k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/120k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/120k/top23/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_top23" "1" "120k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/120k/top23/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/120k/top23/IoTID20/base" "DOMINANT_B4_64_NORM_120k_IoT23_top23" "1" "120k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/120k/top23/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/120k/top23/IoT_traces/base" "DOMINANT_B4_64_NORM_120k_IoT23_top23" "1" "120k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/120k/top23/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/120k/top53/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_top53" "1" "120k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/120k/top53/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/120k/top53/IoTID20/base" "DOMINANT_B4_64_NORM_120k_IoT23_top53" "1" "120k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/120k/top53/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/120k/top53/IoT_traces/base" "DOMINANT_B4_64_NORM_120k_IoT23_top53" "1" "120k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/120k/top53/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/120k/top63/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_top63" "1" "120k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/120k/top63/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/120k/top63/IoTID20/base" "DOMINANT_B4_64_NORM_120k_IoT23_top63" "1" "120k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/120k/top63/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/120k/top63/IoT_traces/base" "DOMINANT_B4_64_NORM_120k_IoT23_top63" "1" "120k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/120k/top63/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/90k/only_etdg/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_only_etdg" "1" "90k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/90k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/90k/only_etdg/IoTID20/base" "DOMINANT_B4_64_NORM_90k_IoT23_only_etdg" "1" "90k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/90k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/90k/only_etdg/IoT_traces/base" "DOMINANT_B4_64_NORM_90k_IoT23_only_etdg" "1" "90k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/90k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/90k/top23/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_top23" "1" "90k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/90k/top23/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/90k/top23/IoTID20/base" "DOMINANT_B4_64_NORM_90k_IoT23_top23" "1" "90k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/90k/top23/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/90k/top23/IoT_traces/base" "DOMINANT_B4_64_NORM_90k_IoT23_top23" "1" "90k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/90k/top23/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/90k/top53/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_top53" "1" "90k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/90k/top53/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/90k/top53/IoTID20/base" "DOMINANT_B4_64_NORM_90k_IoT23_top53" "1" "90k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/90k/top53/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/90k/top53/IoT_traces/base" "DOMINANT_B4_64_NORM_90k_IoT23_top53" "1" "90k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/90k/top53/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/90k/top63/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_top63" "1" "90k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/90k/top63/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/90k/top63/IoTID20/base" "DOMINANT_B4_64_NORM_90k_IoT23_top63" "1" "90k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/90k/top63/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/90k/top63/IoT_traces/base" "DOMINANT_B4_64_NORM_90k_IoT23_top63" "1" "90k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/90k/top63/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/60k/only_etdg/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_only_etdg" "1" "60k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/60k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/60k/only_etdg/IoTID20/base" "DOMINANT_B4_64_NORM_60k_IoT23_only_etdg" "1" "60k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/60k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/60k/only_etdg/IoT_traces/base" "DOMINANT_B4_64_NORM_60k_IoT23_only_etdg" "1" "60k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/60k/only_etdg/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/60k/top23/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_top23" "1" "60k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/60k/top23/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/60k/top23/IoTID20/base" "DOMINANT_B4_64_NORM_60k_IoT23_top23" "1" "60k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/60k/top23/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/60k/top23/IoT_traces/base" "DOMINANT_B4_64_NORM_60k_IoT23_top23" "1" "60k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/60k/top23/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/60k/top53/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_top53" "1" "60k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/60k/top53/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/60k/top53/IoTID20/base" "DOMINANT_B4_64_NORM_60k_IoT23_top53" "1" "60k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/60k/top53/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/60k/top53/IoT_traces/base" "DOMINANT_B4_64_NORM_60k_IoT23_top53" "1" "60k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/60k/top53/IoT23_min_max_benign" "etdg_graph" "IoT_traces"

    #"feature_importance_dataset/60k/top63/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_top63" "1" "60k/base/IoT23_dataset_split_etdg" "feature_importance_dataset/60k/top63/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"feature_importance_dataset/60k/top63/IoTID20/base" "DOMINANT_B4_64_NORM_60k_IoT23_top63" "1" "60k/base/IoTID20_dataset_split_etdg" "feature_importance_dataset/60k/top63/IoT23_min_max_benign" "etdg_graph" "IoTID20"
    #"feature_importance_dataset/60k/top63/IoT_traces/base" "DOMINANT_B4_64_NORM_60k_IoT23_top63" "1" "60k/base/IoT_traces_dataset_split_etdg" "feature_importance_dataset/60k/top63/IoT23_min_max_benign" "etdg_graph" "IoT_traces"



    #"150000/IoT_traces/base" "DOMINANT_B4_64_NORM_150k_IoT23_etdg" "1" "150k/base/IoT_traces_dataset_split_etdg" "150k/IoT23_min_max_benign" "etdg_graph" "IoT_traces"
    #"120000/IoT_traces/base" "DOMINANT_B4_64_NORM_120k_IoT23_etdg" "1" "120k/base/IoT_traces_dataset_split_etdg" "120k/IoT23_min_max_benign/" "etdg_graph" "IoT_traces"
    #"90000/IoT_traces/base" "DOMINANT_B4_64_NORM_90k_IoT23_etdg" "1" "90k/base/IoT_traces_dataset_split_etdg" "90k/IoT23_min_max_benign/" "etdg_graph" "IoT_traces"
    #"60000/IoT_traces/base" "DOMINANT_B4_64_NORM_60k_IoT23_etdg" "1" "60k/base/IoT_traces_dataset_split_etdg" "60k/IoT23_min_max_benign/" "etdg_graph" "IoT_traces"

    #"150000/IoTID20/base" "DOMINANT_B4_64_NORM_150k_IoT23_etdg" "1" "150k/base/IoTID20_dataset_split_etdg" "150k/IoT23_min_max_benign/" "etdg_graph" "IoTID20"
    "120000/IoTID20/base" "DOMINANT_B4_64_NORM_120k_IoT23_etdg" "1" "120k/base/IoTID20_dataset_split_etdg" "120k/IoT23_min_max_benign/" "etdg_graph" "IoTID20"
    #"90000/IoTID20/base" "DOMINANT_B4_64_NORM_90k_IoT23_etdg" "1" "90k/base/IoTID20_dataset_split_etdg" "90k/IoT23_min_max_benign/" "etdg_graph" "IoTID20"
    #"60000/IoTID20/base" "DOMINANT_B4_64_NORM_60k_IoT23_etdg" "1" "60k/base/IoTID20_dataset_split_etdg" "60k/IoT23_min_max_benign/" "etdg_graph" "IoTID20"

    #"150000/IoT_traces/base" "DOMINANT_B4_64_NORM_150k_IoT23_tdg" "1" "150k/base/IoT_traces_dataset_split_etdg" "150k/IoT23_min_max_benign/" "tdg_graph" "IoT_traces"
    #"120000/IoT_traces/base" "DOMINANT_B4_64_NORM_120k_IoT23_tdg" "1" "120k/base/IoT_traces_dataset_split_etdg" "120k/IoT23_min_max_benign/" "tdg_graph" "IoT_traces"
    #"90000/IoT_traces/base" "DOMINANT_B4_64_NORM_90k_IoT23_tdg" "1" "90k/base/IoT_traces_dataset_split_etdg" "90k/IoT23_min_max_benign/" "tdg_graph" "IoT_traces"
    #"60000/IoT_traces/base" "DOMINANT_B4_64_NORM_60k_IoT23_tdg" "1" "60k/base/IoT_traces_dataset_split_etdg" "60k/IoT23_min_max_benign/" "tdg_graph" "IoT_traces"
    
    
    #"150000/IoTID20/base" "DOMINANT_B4_64_NORM_150k_IoT23_tdg" "1" "150k/base/IoTID20_dataset_split_etdg" "150k/IoT23_min_max_benign/" "tdg_graph" "IoTID20"
    #"120000/IoTID20/base" "DOMINANT_B4_64_NORM_120k_IoT23_tdg" "1" "120k/base/IoTID20_dataset_split_etdg" "120k/IoT23_min_max_benign/" "tdg_graph" "IoTID20"
    #"90000/IoTID20/base" "DOMINANT_B4_64_NORM_90k_IoT23_tdg" "1" "90k/base/IoTID20_dataset_split_etdg" "90k/IoT23_min_max_benign/" "tdg_graph" "IoTID20"
    #"60000/IoTID20/base" "DOMINANT_B4_64_NORM_60k_IoT23_tdg" "1" "60k/base/IoTID20_dataset_split_etdg" "60k/IoT23_min_max_benign/" "tdg_graph" "IoTID20"
    
    #"10000/IoTID20/base" "DOMINANT_B4_64_NORM_10k_IoT23_tdg" "1" "10k/base/IoTID20_dataset_split_etdg" "10k/IoT23_min_max_benign/" "tdg_graph" "IoTID20"
    #"10000/IoT_traces/base" "DOMINANT_B4_64_NORM_10k_IoT23_tdg" "1" "10k/base/IoT_traces_dataset_split_etdg" "10k/IoT23_min_max_benign/" "tdg_graph" "IoT_traces"
    #"10000/IoT23/base" "DOMINANT_B4_64_NORM_10k_IoT23_tdg" "1" "10k/base/IoT23_dataset_split_etdg" "10k/IoT23_min_max_benign" "tdg_graph" "IoT23"

    #Parte da lanciare senza il split_test
    #"150000/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_etdg" "1" "150k/base/IoT23_dataset_split_etdg" "150k/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"120000/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_etdg" "1" "120k/base/IoT23_dataset_split_etdg" "120k/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"90000/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_etdg" "1" "90k/base/IoT23_dataset_split_etdg" "90k/IoT23_min_max_benign" "etdg_graph" "IoT23"
    #"60000/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_etdg" "1" "60k/base/IoT23_dataset_split_etdg" "60k/IoT23_min_max_benign" "etdg_graph" "IoT23"
    
    #"150000/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT23_tdg" "1" "150k/base/IoT23_dataset_split_etdg" "150k/IoT23_min_max_benign" "tdg_graph" "IoT23"
    #"120000/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT23_tdg" "1" "120k/base/IoT23_dataset_split_etdg" "120k/IoT23_min_max_benign" "tdg_graph" "IoT23"
    #"90000/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT23_tdg" "1" "90k/base/IoT23_dataset_split_etdg" "90k/IoT23_min_max_benign" "tdg_graph" "IoT23"
    #"60000/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT23_tdg" "1" "60k/base/IoT23_dataset_split_etdg" "60k/IoT23_min_max_benign" "tdg_graph" "IoT23"
    
    

    #Parte Traces
    #"150000/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT_traces_etdg" "1" "150k/base/IoT23_dataset_split_etdg" "150k/IoT_traces_min_max_benign/" "etdg_graph" "IoT23"
    #"120000/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT_traces_etdg" "1" "120k/base/IoT23_dataset_split_etdg" "120k/IoT_traces_min_max_benign/" "etdg_graph" "IoT23"
    #"90000/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT_traces_etdg" "1" "90k/base/IoT23_dataset_split_etdg" "90k/IoT_traces_min_max_benign/" "etdg_graph" "IoT23"
    #"60000/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT_traces_etdg" "1" "60k/base/IoT23_dataset_split_etdg" "60k/IoT_traces_min_max_benign/" "etdg_graph" "IoT23"

    #"150000/IoT23/base" "DOMINANT_B4_64_NORM_150k_IoT_traces_tdg" "1" "150k/base/IoT23_dataset_split_tdg" "150k/IoT_traces_min_max_benign/" "tdg_graph" "IoT23"
    #"120000/IoT23/base" "DOMINANT_B4_64_NORM_120k_IoT_traces_tdg" "1" "120k/base/IoT23_dataset_split_tdg" "120k/IoT_traces_min_max_benign/" "tdg_graph" "IoT23"
    #"90000/IoT23/base" "DOMINANT_B4_64_NORM_90k_IoT_traces_tdg" "1" "90k/base/IoT23_dataset_split_tdg" "90k/IoT_traces_min_max_benign/" "tdg_graph" "IoT23"
    #"60000/IoT23/base" "DOMINANT_B4_64_NORM_60k_IoT_traces_tdg" "1" "60k/base/IoT23_dataset_split_tdg" "60k/IoT_traces_min_max_benign/" "tdg_graph" "IoT23"
    
    #"150000/IoTID20/base" "DOMINANT_B4_64_NORM_150k_IoT_traces_etdg" "1" "150k/base/IoTID20_dataset_split_etdg" "150k/IoT_traces_min_max_benign/" "etdg_graph" "IoTID20"
    #"120000/IoTID20/base" "DOMINANT_B4_64_NORM_120k_IoT_traces_etdg" "1" "120k/base/IoTID20_dataset_split_etdg" "120k/IoT_traces_min_max_benign/" "etdg_graph" "IoTID20"
    #"90000/IoTID20/base" "DOMINANT_B4_64_NORM_90k_IoT_traces_etdg" "1" "90k/base/IoTID20_dataset_split_etdg" "90k/IoT_traces_min_max_benign/" "etdg_graph" "IoTID20"
    #"60000/IoTID20/base" "DOMINANT_B4_64_NORM_60k_IoT_traces_etdg" "1" "60k/base/IoTID20_dataset_split_etdg" "60k/IoT_traces_min_max_benign/" "etdg_graph" "IoTID20"

    #"150000/IoTID20/base" "DOMINANT_B4_64_NORM_150k_IoT_traces_tdg" "1" "150k/base/IoTID20_dataset_split_tdg" "150k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
    #"120000/IoTID20/base" "DOMINANT_B4_64_NORM_120k_IoT_traces_tdg" "1" "120k/base/IoTID20_dataset_split_tdg" "120k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
    #"90000/IoTID20/base" "DOMINANT_B4_64_NORM_90k_IoT_traces_tdg" "1" "90k/base/IoTID20_dataset_split_tdg" "90k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
    #"60000/IoTID20/base" "DOMINANT_B4_64_NORM_60k_IoT_traces_tdg" "1" "60k/base/IoTID20_dataset_split_tdg" "60k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
    
    #"150000/IoTID20/base" "DOMINANT_B4_64_NO_NORM_150k_IoT_traces_tdg" "0" "150k/base/IoTID20_dataset_split_tdg" "150k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
    #"120000/IoTID20/base" "DOMINANT_B4_64_NO_NORM_120k_IoT_traces_tdg" "0" "120k/base/IoTID20_dataset_split_tdg" "120k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
    #"90000/IoTID20/base" "DOMINANT_B4_64_NO_NORM_90k_IoT_traces_tdg" "0" "90k/base/IoTID20_dataset_split_tdg" "90k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
    #"60000/IoTID20/base" "DOMINANT_B4_64_NO_NORM_60k_IoT_traces_tdg" "0" "60k/base/IoTID20_dataset_split_tdg" "60k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
    
    #"10000/IoT23/base" "DOMINANT_B4_64_NORM_10k_IoT_traces_tdg" "1" "10k/base/IoT23_dataset_split_tdg" "10k/IoT_traces_min_max_benign/" "tdg_graph" "IoT23"
    #"10000/IoTID20/base" "DOMINANT_B4_64_NORM_10k_IoT_traces_tdg" "1" "10k/base/IoTID20_dataset_split_tdg" "10k/IoT_traces_min_max_benign/" "tdg_graph" "IoTID20"
)

cd ../model
# Iterate over configurations

for ((i=0; i<${#configurations[@]}; i+=7)); do
    DATA_FOLD="${configurations[i]}"
    MODEL="${configurations[i+1]}"
    NORMALIZE="${configurations[i+2]}"
    JSON_FOLD="${configurations[i+3]}"
    MM="${configurations[i+4]}"
    GRAPH_TYPE="${configurations[i+5]}"
    DATASET="${configurations[i+6]}"

    DATASET_FOLDER="/user/apaolillo/Output_Grafi/$DATA_FOLD"
    JSON_FOLDER="/user/apaolillo/Output_Grafi/split/$JSON_FOLD"
    MIN_MAX="/user/apaolillo/Output_Grafi/$MM"
    #MIN_MAX="/user/apaolillo/Output_Grafi/min_max_benign/$MM"
    CHECKPOINT_FOLDER="/user/apaolillo/checkpoint_dir/dominant/$MODEL/checkpoints"
    THRESHOLD_PATH="/user/apaolillo/checkpoint_dir/dominant/$MODEL/thresholds"
    RESULT_PATH="/user/apaolillo/checkpoint_dir/dominant/$MODEL/y_pred_true"
    
    echo "Work on: $MODEL"
    python test.py --dataset_folder "${DATASET_FOLDER}" --json_folder "${JSON_FOLDER}" --graph_type "${GRAPH_TYPE}" --checkpoint_path "${CHECKPOINT_FOLDER}" --result_path "${RESULT_PATH}" --min_max "${MIN_MAX}" --dataset "${DATASET}" --normalize "${NORMALIZE}" --threshold_path "${THRESHOLD_PATH}"
done