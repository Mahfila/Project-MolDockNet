"""
Main training entry point for MolDockNet.

Usage:
    python train.py --data data/chembl_5ht1b.csv --config configs/config.yaml

The CSV should have at minimum two columns:
    smiles        - SMILES string of each compound
    docking_score - AutoDock Vina / Glide score in kcal/mol
"""

import argparse
import yaml
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data import load_dataset
from models import MolDockNet
from training import Trainer
from utils import plot_training_curves, plot_predicted_vs_actual, plot_enrichment_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Train MolDockNet")
    parser.add_argument("--data",   required=True, help="Path to CSV dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Data ──────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_dataset(
        args.data,
        smiles_col=config["data"]["smiles_col"],
        target_col=config["data"]["target_col"],
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_seed=config["data"]["random_seed"],
        fp_radius=config["featurization"]["fp_radius"],
        fp_nbits=config["featurization"]["fp_nbits"],
    )

    # ── Model ─────────────────────────────────────────────
    model = MolDockNet(
        node_in_dim=config["featurization"]["node_features"] + 10,  # atom one-hot + extra
        edge_in_dim=config["featurization"]["edge_features"] + 3,
        gnn_hidden=config["model"]["gnn_hidden"],
        gnn_layers=config["model"]["gnn_layers"],
        fp_in_dim=config["featurization"]["fp_nbits"],
        fp_hidden=config["model"]["mlp_hidden"],
        dropout=config["model"]["dropout"],
    )
    print(f"\nMolDockNet  |  {model.count_parameters():,} trainable parameters")

    # ── Train ─────────────────────────────────────────────
    trainer = Trainer(model, config, device=args.device)
    history = trainer.fit(train_ds, val_ds)

    # ── Evaluate ──────────────────────────────────────────
    trainer.load_best()
    metrics, preds, labels = trainer.evaluate(test_ds)

    # ── Plots ─────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    plot_training_curves(history, save_path="results/training_curves.png")
    plot_predicted_vs_actual(labels, preds, metrics, save_path="results/pred_vs_actual.png")
    plot_enrichment_curve(labels, preds, save_path="results/enrichment_curve.png")

    print("\n✓ Results saved to results/")


if __name__ == "__main__":
    main()
