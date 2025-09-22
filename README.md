# DIVERSE: Disagreement-Inducing Vector Evolution For Rashomon Set Exploration

---

## 🌟 Overview

**DIVERSE** is a framework for systematically exploring the **Rashomon set** of neural networks — the collection of models that achieve similar accuracy to a reference model while differing in their predictive behavior.  
The method augments pretrained networks with **Feature-wise Linear Modulation (FiLM)** layers and uses **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** to explore a latent modulation space, discovering diverse model variants **without retraining or gradient access**

---

## ✨ Key Ideas

- 🎛 **FiLM-based modulation**: Introduces lightweight FiLM layers into a frozen pretrained model, enabling controlled activation shifts via a latent vector.  
- 🧬 **CMA-ES optimization**: Gradient-free evolutionary search over latent vectors, targeting disagreement while maintaining accuracy.  
- 📊 **Rigorous Rashomon protocol**: Enforces Rashomon membership on a validation set and reports diversity only on a held-out test set.  
- ⚡ **Scalable exploration**: Substantially reduces computational costs compared to retraining-based approaches.

---

## 🚀 Getting Started

### Requirements

**System**
- Tested on **Ubuntu 22.04 LTS**
- Requires a **bash shell** to run experiment scripts (`.sh`)
- Requires an **NVIDIA GPU with CUDA support**  
  (tested on GeForce RTX 4090 with **CUDA 12.4**)
- Windows users may run the code inside **WSL2** or a Linux container  
  (not officially tested)

**Software**
- [Anaconda](https://www.anaconda.com/)
- Python 3.11.7


### Setup
```bash
conda env create -f environment.yml
```
Activate the new environment: ```conda activate diverse```

### Training the Pretrained Models and Generating Z Seeds
First, make the provided bash script executable:
```bash
chmod +x init.sh
```
Then run it: 
```bash
./init.sh
```

This will:
1. Train the reference (pretrained) models for each dataset.
2. Generate initial latent vectors (**Z seeds**) used for CMA-ES exploration.


## 🏁 Running Baselines
Before running any baselines, ensure the conda environment is activated:
```bash
conda activate diverse
```
### Dropout (Hsu et al., 2024)
We provide an implementation of the dropout-based Rashomon exploration method described in [Hsu et al. (ICLR 2024).](https://proceedings.iclr.cc/paper_files/paper/2024/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)


For MNIST:
```bash
python -m baselines.dropout --method=gaussian --model=mnist --epsilon=0.05 --search_budget=167
```

For ResNet50:
```bash
python -m baselines.dropout --method=gaussian --model=resnet --epsilon=0.05 --search_budget=167
```

For VGG16:
```bash
python -m baselines.dropout --method=gaussian --model=vgg --epsilon=0.05 --search_budget=167
```


### Retraining
**Warning:** Retraining is computationally expensive and may require significant time and GPU resources.You can adjust the number of models to train and evaluate via the ```--search_budget``` flags.
Supported values: **167, 320, 640, 1284, 2562, 5120**.
#### Training
For MNIST:
```bash
python -m baselines.retraining --model=mnist --start_seed=42 --search_budget=5120
```

For ResNet50:
```bash
python -m baselines.retraining --model=resnet --start_seed=42 --search_budget=640
```

For VGG16:
```bash
python -m baselines.retraining --model=vgg --start_seed=45 --search_budget=640
```

#### Evaluation
After training, results can be evaluated on the test set.
Outputs will be stored in 5 folders (```epsilon_0.01```, ```epsilon_0.02```, ```epsilon_0.03```, ```epsilon_0.04```, ```epsilon_0.05```) in the following path:
```
baseline_evaluations/retraining/retraining_<model>/epsilon_<epsilon_value>/
```

For MNIST:
```bash
python -m baselines.retraining_evaluator --model=mnist --search_budget=5120
```

For ResNet50:
```bash
python -m baselines.retraining_evaluator --model=resnet --search_budget=640
```

For VGG16:
```bash
python -m baselines.retraining_evaluator --model=vgg --search_budget=640
```

