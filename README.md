# An-Improved-Point-Level-Supervision-Method-for-TAL
This is the offical pyTorch implementation of our IEEE Access 2023 paper

### Requirement

- python 3.6+
- pytorch 1.6+
- CUDA 10.2+
- tensorboard-logger
- tensorboard
- scipy
- pandas 
- joblib
- tqdm

### Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n IPLSM python=3.* -y
conda activate IPLSM
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
# CUDA 10.2
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 –f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.3
- pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Data Preparation
- Download extract features (THUMOS14) from [LACP](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points).
- Place the features inside the dataset folder.

### Testing
```shell
conda activate IPLSM
python main_eval.py
```
### References
We referenced the repo below for the code.
- [LACP](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points).
