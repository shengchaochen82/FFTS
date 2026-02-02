# Federated Foundation Models on Heterogeneous Time Series (FFTS) - AAAI'2025

[![arXiv](https://img.shields.io/badge/arXiv-2412.08906-b31b1b.svg)](https://arxiv.org/abs/2412.08906)
[![AAAI](https://img.shields.io/badge/AAAI-2025-1f4b99.svg)](https://aaai.org/aaai-conference/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)

Official implementation for the AAAI'25 paper "Federated Foundation Models on Heterogeneous Time Series".

---

## Overview
Training general-purpose time series foundation models across diverse domains is challenging due to severe statistical heterogeneity. FFTS tackles this with a federated learning formulation where each dataset owner is a client with its own local model. Client-side and server-side regularization aligns shared knowledge across heterogeneous datasets while preserving domain-specific patterns. The resulting foundation model generalizes well across forecasting, imputation, and anomaly detection tasks.

---

## Highlights
- Federated foundation model training across heterogeneous time series datasets.
- Client-specific local models with shared knowledge alignment.
- Unified adaptation architecture for downstream tasks.
- Learnable time-scale weights introduced in the ATM module.

---

## News
:triangular_flag_on_post: **2025.08** Codebase restructured. **New:** learnable time-scale weights in the ATM module.

:triangular_flag_on_post: **2025.01** Pretraining datasets available via [Monash Time Series Repo](https://forecastingdata.org/); see preprocessing helpers in `preprocessing.ipynb`.

:triangular_flag_on_post: **2024.12** Preprint posted on arXiv: [2412.08906](https://arxiv.org/abs/2412.08906).

:triangular_flag_on_post: **2024.12** Accepted at **AAAI 2025**.

---

## Method At a Glance
![What's News](assest/difference.png "FFTS overview")

## Pretraining Datasets
![Datasets](assest/pretrain_data.png "Pretraining datasets")

## Unified Adaptation Architecture
![Adaption](assest/adaption.png "Adaptation architecture")

---

## Repository Structure
```text
.
├─ data_provider/       # Dataset loading and preprocessing
├─ flcore/              # Federated learning servers and clients
├─ models/              # FFTS models and components
├─ utils/               # Utilities and helpers
├─ preprocessing.ipynb  # Unified preprocessing notebook
└─ main.py              # Training entry point
```

---

## Quickstart
### 1) Prepare data
- Download datasets from the [Monash Time Series Repo](https://forecastingdata.org/).
- Follow the preprocessing steps in `preprocessing.ipynb` to unify formats.

### 2) Run a training job
The main entry point is `main.py`. The following example includes all required flags and uses default values where applicable:

```bash
python main.py \
  --task pretrain \
  --task_note demo \
  --is_training 1 \
  --algorithm FFTS \
  --dataset weather
```

### 3) Explore configuration
See the full argument list in `main.py`, including:
- `--task`: task type (pretrain, forecasting, imputation, anomaly detection, etc.)
- `--dataset`: dataset name
- `--algorithm`: federated algorithm (e.g., `FFTS`, `FedAvg`)
- `--global_rounds`, `--local_epochs`, `--batch_size`: training settings

---

## Roadmap
- [x] Release code
- [x] Release detailed training tutorials
- [x] Pretraining data download and tutorials
- [x] Release paper

---

## Abstract
Training general-purpose time series foundation models with robust generalization capabilities across diverse applications remains an open challenge. Existing approaches often fuse cross-domain time series datasets to extract shared subsequences as tokens for Transformer-based models, but severe statistical heterogeneity limits their effectiveness. **To tackle this challenge, FFTS proposes a federated learning approach for heterogeneous time series foundation model training.** Each data-holding organization is treated as an independent client in a collaborative federated setting, enabling client-specific local models to preserve dataset-specific characteristics. A new regularization mechanism is applied on both client and server to align shared knowledge across heterogeneous datasets. Extensive experiments on benchmark datasets demonstrate the effectiveness of this approach, with strong generalization across forecasting, imputation, and anomaly detection tasks.

---

## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{chen2025federated,
  title={Federated foundation models on heterogeneous time series},
  author={Chen, Shengchao and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={15},
  pages={15839--15847},
  year={2025}
}
```

---

## Acknowledgement
> [!note]
> We are reshaping the codebase; some interfaces may change.
> We are grateful for the many excellent open source frameworks that have given us support, including [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [PFLlib](https://github.com/TsingZ0/PFLlib).
