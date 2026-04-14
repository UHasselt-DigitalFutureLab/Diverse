# DIVERSE: Disagreement-Inducing Vector Evolution For Rashomon Set Exploration [ICLR 2026]

<p align="center">
  <strong>
    <a href="https://be.linkedin.com/in/gilles-eerlings" target="_blank">Gilles Eerlings</a><sup>1,2,3</sup> &nbsp;·&nbsp;
    <a href="https://brent-zoomers.github.io/" target="_blank">Brent Zoomers</a><sup>1,2,4</sup> &nbsp;·&nbsp;
    <a href="https://www.uhasselt.be/en/who-is-who/jori-liesenborgs" target="_blank">Jori Liesenborgs</a><sup>1,2</sup> &nbsp;·&nbsp;
    <a href="https://gustavorovelo.net/" target="_blank">Gustavo Rovelo Ruiz</a><sup>1,2</sup> &nbsp;·&nbsp;
    <a href="https://krisluyten.net/" target="_blank">Kris Luyten</a><sup>1,2,3</sup>
  </strong>
</p>

<p align="center">
  <sup>1</sup> UHasselt Digital Future Lab &nbsp;·&nbsp;
  <sup>2</sup> Flanders Make &nbsp;·&nbsp;
  <sup>3</sup> Flanders AI Research &nbsp;·&nbsp;
  <sup>4</sup> FWO (Fonds Wetenschappelijk Onderzoek – Vlaanderen)
</p>
<br>
<p align="center">
  <a href="https://iclr.cc/virtual/2026/poster/10007789"><img src="https://img.shields.io/badge/ICLR-2026-4B44CE" alt="ICLR"></a>
  &nbsp;
  <a href="https://openreview.net/forum?id=kQjSUHC84V"><img src="https://img.shields.io/badge/OpenReview-kQjSUHC84V-8C1B13"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2601.20627"><img src="https://img.shields.io/badge/arXiv-2601.20627-B31B1B?logo=arxiv&logoColor=white"></a>
  &nbsp;
  <a href="https://polyformproject.org/licenses/noncommercial/1.0.0/"><img src="https://img.shields.io/badge/license-PolyForm--NC-blue"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Environment-Anaconda-44A833?logo=anaconda&logoColor=white">
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.12.9-3776AB?logo=python&logoColor=white">
  &nbsp;
  <img src="https://img.shields.io/badge/Tested%20on-Ubuntu%2022.04-E95420?logo=ubuntu&logoColor=white">
</p>

## 🌟 Overview

**DIVERSE** is a framework for systematically exploring the **Rashomon set** of neural networks, which is the collection of models that achieve similar accuracy to a reference model while differing in their predictive behavior.  
The method augments pretrained networks with **Feature-wise Linear Modulation (FiLM)** layers and uses **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** to explore a latent modulation space, discovering diverse model variants **without retraining or requiring gradient access**.

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
- Python 3.12.9


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

## Running CMA-ES Search
Before running, ensure the conda environment is activated:
```bash
conda activate diverse
```
This script performs an extensive hyperparameter sweep, which can take a long time and heavily use the GPU. Parallelism is controlled through subprocesses; to adjust the number of workers, edit ```utils/experiment_parameters.py```

You have to repeat the following command for each epsilon and each model type (exception for the vision transformer):
```bash
python run_epsilon_CMA.py --model_type=<model_type> --epsilon=<epsilon>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for run_epsilon_CMA.py</span></summary>

| Argument | Type | Required | Options | Description |
| --- | --- | --- | --- | ---|
| `--model_type` | str | ✅ | mnist, resnet50_pneumonia, vgg16_cifar10, vision_transformer_cifar10 | Type of model to use |
| `--epsilon` | float | ✅ | 0.01, 0.02, 0.03, 0.04, 0.05 | Epsilon value for the Rashomon requirement |
| `--lambda_val` | float | ❌ | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 (default), 0.6, 0.7, 0.8, 0.9, 1.0 | Lambda value used as a mixing weight for hard and soft disagreement|

</details>

### Evaluating CMA-ES Results
Once you have run all CMA-ES runs, you can evaluate each epsilon, z dimension and dataset combination with the following:
```bash
python -m CMA.CMA_evaluation --model_type=<model_type> --epsilon=<epsilon> --z_dim=<z_dim>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for CMA.CMA_evaluation.py</span></summary>

| Argument | Type | Required | Options | Description |
| --- | --- | --- | --- | ---|
| `--model_type` | str | ✅ | mnist, resnet50_pneumonia, vgg16_cifar10, vision_transformer_cifar10 | Type of model to use |
| `--epsilon` | float | ✅ | 0.01, 0.02, 0.03, 0.04, 0.05 | Epsilon value for the Rashomon requirement |
| `--z_dim` | int | ✅ | 2, 4, 8, 16, 32, 64 | Dimension of the z vector |
| `--lambda_val` | float | ❌ | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 (default), 0.6, 0.7, 0.8, 0.9, 1.0 | Lambda value used as a mixing weight for hard and soft disagreement|

</details>

## 🏁 Running Baselines
Before running any baselines, ensure the conda environment is activated:
```bash
conda activate diverse
```
### Dropout (Hsu et al., 2024)
We provide an implementation of the dropout-based Rashomon exploration method described in [Hsu et al. (ICLR 2024).](https://proceedings.iclr.cc/paper_files/paper/2024/file/8cd1ce03ea58b3d7dfd809e4d42f08ea-Paper-Conference.pdf)


```bash
python -m baselines.dropout --model=<model_type> --epsilon=<epsilon> --search_budget=<search_budget>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for baselines.dropout.py</span></summary>

| Argument | Type | Required | Options | Description |
| --- | --- | --- | --- | ---|
| `--model` | str | ✅ | mnist, resnet50_pneumonia, vgg16_cifar10 | Type of model to use |
| `--epsilon` | float | ✅ | 0.01, 0.02, 0.03, 0.04, 0.05 | Epsilon value for the Rashomon requirement |
| `--search_budget` | int | ✅ | 162, 320, 640, 1284, 2562, 5120 | Total number of models to evaluate (approx). |

</details>



### Retraining
**Warning:** Retraining is computationally expensive and may require significant time and GPU resources.
#### Training
```bash
python -m baselines.retraining --model=<model> --start_seed=<start_seed> --search_budget=<search_budget>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for baselines.retraining.py</span></summary>

| Argument | Type | Required | Options | Description |
| --- | --- | --- | --- | ---|
| `--model` | str | ✅ | mnist, resnet, vgg | Type of model to use |
| `--start_seed` | int | ✅ | 42, 45 | Starting seed value, vgg requires seed 45 |
| `--search_budget` | int | ✅ | 162, 320, 640, 1284, 2562, 5120 | Total number of models to evaluate (approx). |

</details>

#### Evaluation
After training, results can be evaluated on the test set.
Outputs will be stored in a folder in the following path: `baseline_evaluations/retraining/retraining_<model>/epsilon_<epsilon_value>/`

```bash
python -m baselines.retraining_evaluator --model=<model> --epsilon=<epsilon> --search_budget=<search_budget>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for baselines.retraining_evaluator.py</span></summary>

| Argument | Type | Required | Options | Description |
| --- | --- | --- | --- | ---|
| `--model` | str | ✅ | mnist, resnet, vgg | Type of model to use |
| `--epsilon` | float | ✅ | 0.01, 0.02, 0.03, 0.04, 0.05 | Epsilon value for the Rashomon requirement |
| `--search_budget` | int | ✅ | 162, 320, 640, 1284, 2562, 5120 | Total number of models to evaluate should be the same as the budget used while training. |

</details>

## Plotting the results
To plot the results, you will first have to run each CMA search and evaluation for every dataset and epsilon, and have also run and evaluate all baselines for each dataset on the same epsilons.

Once the results are available, generate the plots with:
```bash
python -m utils.plotter
```

## Citation

```bibtex
@inproceedings{
  eerlings2026diverse,
  title={{DIVERSE}: Disagreement-Inducing Vector Evolution for Rashomon Set Exploration},
  author={Gilles Eerlings and Brent Zoomers and Jori Liesenborgs and Gustavo Rovelo Ruiz and Kris Luyten},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=kQjSUHC84V}
}
```
