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
Download dataset snaposhots and stats by following instructions reported [here]().</br>
**NOTE** The link to the dataset will be released after the acceptance of the paper.


## Download Dynamic Graphs Dependencies
```bash
mkdir gnn-network-analysis/dynamic_graphs
cd gnn-network-analysis/dynamic_graphs
git clone https://github.com/ciccio42/EvolveGCN.git
git checkout pub_iot
```

## Train and Test
```bash
cd DOMINANT
# Dominant Train
nohup train.sh > dominant_train.txt &

# Dominant Test
nohup test_tdg.sh > test_tdg.txt & # Test tdg model
nohup test_etdg.sh > test_etdg.txt & # Test e-tdg model
```

```bash
cd OCGNN
# OC-GNN Train
nohup train.sh > dominant_train.txt &

# OC-GNN Test
nohup test_tdg.sh > test_tdg.txt & # Test tdg model
nohup test_etdg.sh > test_etdg.txt & # Test e-tdg model
```


**NOTE** Configure the bash file correctly. You need to set the snapshot to use and your paths to dataset.

## Note
<For any errors and/or questions about the code either open an issue or mail **frosa@unisa.it**, with object "QUESTION-CODE: Graph Neural Networks for IoT Security: A Comparative Study">


