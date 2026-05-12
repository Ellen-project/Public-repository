<div align="center">
  
  # Neuromorphic AI porject.
  
A hybrid AI under development, inspired by neurons and synapses in the brain using SNN-based architecture.
</div>

## Usage

### 1. Required Dependencies
- Python 3.12.9 (or lower if needed)
- CUDA Toolkit 12.8 (or lower if needed)
- MSVC 2022
- NVCC

### 2. Install Python Packages
Install the required Python packages using:

```
pip3 install -r requirements.txt
```
## Build Kernel
```
python src/main.py --mode train_test --run-train --data modelData/trainData/shortTrain.txt --model-dir modelData/tokenizer --seq-len 8 --batch-size 1 --hidden1 32 --hidden2 32 --tbptt-len 4 --epochs 1 
```

### ⚠️ If you get an error log
```
ERROR: Could not find a version that satisfies the requirement torch==2.8.0+cu129 (from versions: 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.7.1, 2.8.0)
```
visit [PyTorch](https://pytorch.org/) website and install the Stable (2.8.0) version with CUDA 12.8.
