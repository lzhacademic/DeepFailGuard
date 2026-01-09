# DeepFailGuard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official implementation of the paper: **"DeepFailGuard: Failure Prediction based Fault Tolerance Approach for Deep Neural Network"**.

---

## 📌 Overview
DeepFailGuard is a fault tolerance approach designed to enhance the reliability of DNN systems by predicting the failure probabilities of each version. This repository provides the complete pipeline, including fault injection, model training, distance estimation, evaluation and so on.


## 📂 Project Structure
```text
DeepFailGuard/
├── dfg_results/          # Raw experimental results and statistical analysis (Summary of results/)
├── envs_requirements/    # Environment configuration files (.yml and .txt)
├── mydataset/            # Code for dataset processing (Data files should be placed here)
├── mymodel/              # Code for model architectures (Pre-trained weights should be placed here)
├── input_information/    # [To be created] Cached boundary distance information
├── Attacker.py           # Implementation of adversarial pertubation analysis
├── DatasetManager.py     # Dataset creation and management
├── DeepFailGuard.py      # Core implementation of the DeepFailGuard method
├── FaultInjector.py      # Module for injecting faults into datasets and models
├── ModelManager.py       # Model creation and management
├── ModelTrainer.py       # Deep learning model trainer
├── utils.py              # Utility functions
├── main_train_model.py   # Entrance for training and evaluating DL models
├── main_train_DeepFailGuard.py     # Entrance for offline training process of DeepFailGuard
├── main_cache_test_distances.py    # Entrance for pre-calculating and caching boundary distances
├── main_test_DeepFailGuard.py      # Entrance for real-time evaluation of DeepFailGuard
└── main_test_DeepFailGuard_from_cached.py  # Entrance for fast evaluation using cached data
```

## ⚙️ Environment Setup
We provide two ways to configure the running environment located in `envs_requirements/`:
1. Using Conda (Recommended):
```bash
conda env create -f envs_requirements/dl_environment.yml
conda activate dl
```
2. Using pip:
```bash
pip install -r envs_requirements/dl_requirements.txt
```

## 📊 Dataset
Due to storage limits, large files are hosted on **[Google Drive](https://drive.google.com/drive/folders/1EjoRljtQvErp5v36hrR9DhRJpK7shoFJ?usp=sharing)**.

1. **Datasets**
+ Download [`mydatasets.zip`](https://drive.google.com/drive/folders/1EjoRljtQvErp5v36hrR9DhRJpK7shoFJ?usp=sharing) and extract the `data/` folder into the [`mydataset/`](https://github.com/lzhacademic/DeepFailGuard/tree/main/mydataset) directory.
+ This includes 7 different datasets used in our study.
2. **Models and Weights**
+ Download [`mymodel.zip`](https://drive.google.com/drive/folders/1EjoRljtQvErp5v36hrR9DhRJpK7shoFJ?usp=sharing) and extract the `model/` folder into the [`mymodel/`](https://github.com/lzhacademic/DeepFailGuard/tree/main/mymodel) directory.
+ We provide 5 architectures (ResNet, AlexNet, DenseNet, SqueezeNet, VGG) with injected faults.
3. **Cached Information (For Fast Evaluation)**
+ Download [`input_information.zip`](https://drive.google.com/drive/folders/1EjoRljtQvErp5v36hrR9DhRJpK7shoFJ?usp=sharing) and extract the `input_information/` folder directly into the project root [`DeepFailGuard/`](https://github.com/lzhacademic/DeepFailGuard/tree/main).
+ This folder contrains pre-calculated boundary distances to accelerate evaluation.

## 🚀 Usage and Reproduction
**Step 1: Model Training and DeepFailGuard Module Training(Optional)**
If you wish to train additional models or use your own architectures:
1. Modify `ModelManager.py`.
2. Run `python main_train_model.py`.
3. Run `python main_train_DeepFailGuard.py` for offline training.

**Step 2: Evaluation (Reproduce Paper Results)**
There are two ways to evaluate DeepFailGuard:
+ **Method A: Fast Evaluation using cached Data (Recommended)** This method uses pre-calculated distances to save time.
```bash
python main_test_DeepFailGuard_from_cached.py
```
+ **Method B: Real-time Evaluation** This method calculates information on-the-fly without using local caches.
```bash
python main_test_DeepFailGuard.py
```

**Step 3: Results Analysis**
The raw results and statistical summaries used in the paper can be found in [`dfg_results/Summary of results/`](https://github.com/lzhacademic/DeepFailGuard/tree/main/dfg_results/Summary%20of%20results). 

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

## 📧 Contact
For any questions or bug reports, please contact: `lzh123698745@buaa.edu.cn`.