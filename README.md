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

