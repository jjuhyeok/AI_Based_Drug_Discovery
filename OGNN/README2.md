# OGNN_mixup
----------------------------------------------
Operating System: Linux 5.15.0-131-generic
Python Version: 3.10.13
CPU: x86_64
CPU Cores: 16 physical, 32 logical
Total Memory: 251.51 GB
Available Memory: 175.05 GB
Number of GPUs: 4
GPU 0: NVIDIA RTX A6000
  - CUDA Capability: (8, 6)
  - Total Memory: 47.43 GB
GPU 1: NVIDIA RTX A6000
  - CUDA Capability: (8, 6)
  - Total Memory: 47.43 GB
GPU 2: NVIDIA RTX A6000
  - CUDA Capability: (8, 6)
  - Total Memory: 47.43 GB
GPU 3: NVIDIA RTX A6000
  - CUDA Capability: (8, 6)
  - Total Memory: 47.43 GB
----------------------------------------------
## **Setup environment**
    cd OGNN
    conda env create -f environment.yml
    conda activate OGNN
    pip install dgl==1.1.2
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


### Step1 : prepare finetune
    python prepare_finetune.py

### Step2 : finetune_mixup (fold 1부터 5까지 실행)
    python finetune_mixup.py \
        --fold 1 \
        --seed 724

### Step3 : test_prediction
    python test_prediction.py
