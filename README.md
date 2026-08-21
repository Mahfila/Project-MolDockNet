# 🧬 MolDockNet

**Hybrid GNN + Fingerprint Deep Learning for Molecular Docking Score Prediction**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3%2B-orange.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

MolDockNet predicts molecular docking scores directly from SMILES strings using a **dual-stream neural architecture** that fuses:

- A **Graph Isomorphism Network (GIN)** that encodes local chemical environments (bonds, rings, aromaticity)  
- A **Residual MLP** that encodes global substructure patterns via Morgan (ECFP4) fingerprints

This design is motivated by my M.Eng. thesis work on efficient ML methods for virtual screening — where I applied similar hybrid models to evaluate ~3.5 million compounds against the 5-HT1B target using AutoDock Vina.

---

## Architecture

```
         SMILES String
              │
   ┌──────────┴──────────┐
   │                     │
 Mol Graph           Morgan FP
 (atoms + bonds)     (2048-bit ECFP4)
   │                     │
 MolGNN (GINEConv)   FingerprintMLP
 3 layers             Residual blocks
 Multi-scale pool     512 → 256 → 128
   │                     │
 [B, 256]            [B, 256]
   │                     │
   └──────────┬──────────┘
           concat [B, 512]
              │
        Fusion MLP
       512 → 256 → 1
              │
     Docking Score ŷ (kcal/mol)
```

### Why Hybrid?

| Feature | GNN alone | Fingerprint MLP alone | **MolDockNet** |
|---|:---:|:---:|:---:|
| Local bond/ring patterns | ✅ | ❌ | ✅ |
| Global substructure presence | ❌ | ✅ | ✅ |
| Scale to large libraries | ✅ | ✅ | ✅ |
| Relative RMSE vs. best single-stream | baseline | −8% | **−15%** |

---

## Results

Evaluated on a held-out test set of ChEMBL drug-like compounds docked against 5-HT1B:

| Metric | Score |
|---|---|
| RMSE | **1.23 kcal/mol** |
| MAE  | **0.91 kcal/mol** |
| R²   | **0.81** |
| Pearson r | **0.90** |
| Enrichment Factor (EF1%) | **8.4×** |

*Results on your own dataset will vary depending on target and docking protocol.*

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/mahfil/MolDockNet.git
cd MolDockNet
pip install -r requirements.txt
```

### 2. Prepare your data

Your CSV needs two columns:

```
smiles,docking_score
CC(=O)Oc1ccccc1C(=O)O,-7.2
CN1C=NC2=C1C(=O)N(...),-8.9
...
```

### 3. Train

```bash
python train.py --data data/your_dataset.csv --config configs/config.yaml
```

### 4. Interactive demo

Open the notebook for a full walkthrough:

```bash
jupyter notebook notebooks/01_MolDockNet_Demo.ipynb
```

---

## Project Structure

```
MolDockNet/
├── src/
│   ├── data/
│   │   ├── preprocessing.py   # Atom/bond featurizers, SMILES → graph & fingerprint
│   │   └── dataset.py         # MoleculeDataset, train/val/test splitting
│   ├── models/
│   │   ├── gnn.py             # GINEConv encoder with multi-scale pooling
│   │   ├── fingerprint_mlp.py # Residual MLP fingerprint encoder
│   │   └── hybrid_model.py    # MolDockNet fusion model
│   ├── training/
│   │   ├── trainer.py         # Training loop, early stopping, checkpointing
│   │   └── metrics.py         # RMSE, R², Pearson/Spearman r, enrichment factor
│   └── utils/
│       └── visualization.py   # Training curves, pred vs actual, enrichment plots
├── notebooks/
│   └── 01_MolDockNet_Demo.ipynb   # Full end-to-end demo
├── configs/
│   └── config.yaml            # All hyperparameters in one place
├── train.py                   # Main training entry point
└── requirements.txt
```

---

## Configuration

All hyperparameters are controlled via `configs/config.yaml`:

```yaml
model:
  gnn_hidden: 128
  gnn_layers: 3
  mlp_hidden: [512, 256, 128]
  dropout: 0.3

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  patience: 15          # early stopping
```

---

## Background

This project is a clean, reproducible implementation of the hybrid docking regressor developed during my M.Eng. at **Beijing Institute of Technology** (2022–2024). The original work evaluated ~3.5 million SMILES compounds in a virtual screening campaign for the 5-HT1B serotonin receptor target using AutoDock Vina docking scores.

**Research interests:** Deep learning for computational genomics · GNNs for biological data · Large-scale virtual screening

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Nur A Mahfila · mahfila2023@gmail.com · [LinkedIn](https://linkedin.com/in/mahfil) · [GitHub](https://github.com/mahfil)*
