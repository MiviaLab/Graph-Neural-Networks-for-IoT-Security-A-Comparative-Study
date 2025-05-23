# Repository
The code available in this repository has been used for producing the results reported in *Graph Neural Networks for IoT Security: A Comparative Study*

# How to use the code

## Create Conda Environment
```bash
conda create -n anomaly_detection python=3.9
conda activate anomaly_detection
pip install -r requirements.txt
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset Download
```bash
mkdir anomaly_detection_dataset
```
Download dataset snaposhots and stats by following instructions reported [here](https://zenodo.org/records/15181384). </br>


## Download Dynamic Graphs Dependencies
```bash
mkdir gnn-network-analysis/dynamic_graphs
cd gnn-network-analysis/dynamic_graphs
git clone https://github.com/ciccio42/EvolveGCN.git
git checkout compute_roc
```

## Running 


## Note
For any errors and/or questions about the code either open an issue or mail **frosa@unisa.it**, with object "QUESTION-CODE: Graph Neural Networks for IoT Security: A Comparative Study"


