#!/usr/bin/env python
"""
Ablation Study Script for Contrastive VQ Graph Model
This script runs comprehensive ablation studies to reproduce SOTA results
"""

import os
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path

def run_experiment(config, output_dir):
    """Run a single experiment with given configuration"""
    cmd = [
        "python", "train_contrastive_vq.py",
        "--dataset", config["dataset"],
        "--data_path", "./data",
        "--output_path", output_dir,
        "--num_exp", str(config.get("num_exp", 5)),
        "--seed", str(config.get("seed", 0)),
        "--device", str(config.get("device", 0)),

        # Model architecture
        "--num_gnn_layers", str(config["num_gnn_layers"]),
        "--gnn_hidden_dim", str(config["gnn_hidden_dim"]),
        "--gnn_output_dim", str(config["gnn_output_dim"]),
        "--vq_dim", str(config["vq_dim"]),
        "--codebook_size1", str(config["codebook_size1"]),
        "--codebook_size2", str(config["codebook_size2"]),
        "--projection_head_hidden_dim", str(config["projection_head_hidden_dim"]),
        "--projection_head_output_dim", str(config["projection_head_output_dim"]),

        # Training parameters
        "--learning_rate", str(config["learning_rate"]),
        "--weight_decay", str(config.get("weight_decay", 1e-5)),
        "--max_epoch", str(config["max_epoch"]),
        "--patience", str(config.get("patience", 50)),
        "--batch_size", str(config.get("batch_size", 256)),

        # Contrastive learning
        "--contrastive_temp", str(config["contrastive_temp"]),
        "--lambda_commit_loss", str(config["lambda_commit_loss"]),
        "--vq_commitment_weight", str(config["vq_commitment_weight"]),
        "--vq_decay", str(config.get("vq_decay", 0.99)),

        # Other parameters
        "--dropout_ratio", str(config.get("dropout_ratio", 0.1)),
        "--norm_type", config.get("norm_type", "batch"),
        "--activation_gnn", config.get("activation_gnn", "relu"),
        "--num_linear_probe_runs", str(config.get("num_linear_probe_runs", 5)),
        "--output_json", os.path.join(output_dir, f"{config['name']}_results.json")
    ]

    # Add boolean flags
    if config.get("vq_use_cosine_sim", True):
        cmd.append("--vq_use_cosine_sim")
    if config.get("vq_kmeans_init", True):
        cmd.append("--vq_kmeans_init")

    print(f"\n{'='*60}")
    print(f"Running experiment: {config['name']}")
    print(f"Dataset: {config['dataset']}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error in experiment {config['name']}:")
        print(result.stderr)
        return False

    print(f"Experiment {config['name']} completed successfully")
    return True

def get_sota_configs():
    """Get SOTA configurations for different datasets"""
    base_config = {
        "num_exp": 3,
        "num_linear_probe_runs": 5,
        "vq_use_cosine_sim": True,
        "vq_kmeans_init": True,
        "vq_decay": 0.99,
        "weight_decay": 1e-5,
        "dropout_ratio": 0.1,
        "norm_type": "batch",
        "activation_gnn": "relu",
    }

    sota_configs = {
        # Best configurations for each dataset based on your experiments
        "cora": {
            **base_config,
            "name": "cora_sota",
            "dataset": "cora",
            "num_gnn_layers": 2,
            "gnn_hidden_dim": 512,
            "gnn_output_dim": 512,
            "vq_dim": 512,
            "codebook_size1": 256,
            "codebook_size2": 256,
            "projection_head_hidden_dim": 128,
            "projection_head_output_dim": 64,
            "learning_rate": 0.003,
            "max_epoch": 600,
            "patience": 20,
            "batch_size": 256,
            "contrastive_temp": 0.1,
            "lambda_commit_loss": 0.1,
            "vq_commitment_weight": 0.1,
        },

        "citeseer": {
            **base_config,
            "name": "citeseer_sota",
            "dataset": "citeseer",
            "num_gnn_layers": 2,
            "gnn_hidden_dim": 512,
            "gnn_output_dim": 512,
            "vq_dim": 512,
            "codebook_size1": 256,
            "codebook_size2": 256,
            "projection_head_hidden_dim": 128,
            "projection_head_output_dim": 64,
            "learning_rate": 0.002,
            "max_epoch": 600,
            "patience": 20,
            "batch_size": 256,
            "contrastive_temp": 0.1,
            "lambda_commit_loss": 0.1,
            "vq_commitment_weight": 0.1,
        },

        "pubmed": {
            **base_config,
            "name": "pubmed_sota",
            "dataset": "pubmed",
            "num_gnn_layers": 2,
            "gnn_hidden_dim": 1024,
            "gnn_output_dim": 1024,
            "vq_dim": 1024,
            "codebook_size1": 1024,
            "codebook_size2": 1024,
            "projection_head_hidden_dim": 128,
            "projection_head_output_dim": 64,
            "learning_rate": 0.0005,
            "max_epoch": 600,
            "patience": 20,
            "batch_size": 512,
            "contrastive_temp": 0.2,
            "lambda_commit_loss": 0.1,
            "vq_commitment_weight": 0.1,
        },

        "ogbn-arxiv": {
            **base_config,
            "name": "ogbn_arxiv_sota",
            "dataset": "ogbn-arxiv",
            "num_gnn_layers": 3,
            "gnn_hidden_dim": 512,
            "gnn_output_dim": 512,
            "vq_dim": 512,
            "codebook_size1": 1024,
            "codebook_size2": 1024,
            "projection_head_hidden_dim": 256,
            "projection_head_output_dim": 128,
            "learning_rate": 0.001,
            "max_epoch": 1200,
            "patience": 50,
            "batch_size": 2560,
            "contrastive_temp": 0.2,
            "lambda_commit_loss": 0.1,
            "vq_commitment_weight": 0.1,
        },

        "ogbn-products": {
            **base_config,
            "name": "ogbn_products_sota",
            "dataset": "ogbn-products",
            "num_gnn_layers": 3,
            "gnn_hidden_dim": 512,
            "gnn_output_dim": 512,
            "vq_dim": 512,
            "codebook_size1": 2048,
            "codebook_size2": 2048,
            "projection_head_hidden_dim": 256,
            "projection_head_output_dim": 128,
            "learning_rate": 0.001,
            "max_epoch": 1200,
            "patience": 50,
            "batch_size": 5120,
            "contrastive_temp": 0.2,
            "lambda_commit_loss": 0.1,
            "vq_commitment_weight": 0.1,
        },
    }

    return sota_configs

def get_ablation_configs(dataset="cora"):
    """Get ablation study configurations"""
    base = get_sota_configs()[dataset].copy()

    ablation_configs = []

    # Ablation 1: Codebook size
    for cb_size in [128, 256, 512, 1024]:
        config = base.copy()
        config["name"] = f"{dataset}_codebook_{cb_size}"
        config["codebook_size1"] = cb_size
        config["codebook_size2"] = cb_size
        ablation_configs.append(config)

    # Ablation 2: VQ dimension
    for vq_dim in [128, 256, 512, 1024]:
        config = base.copy()
        config["name"] = f"{dataset}_vqdim_{vq_dim}"
        config["vq_dim"] = vq_dim
        config["gnn_output_dim"] = vq_dim
        ablation_configs.append(config)

    # Ablation 3: Number of GNN layers
    for num_layers in [1, 2, 3, 4]:
        config = base.copy()
        config["name"] = f"{dataset}_layers_{num_layers}"
        config["num_gnn_layers"] = num_layers
        ablation_configs.append(config)

    # Ablation 4: Temperature
    for temp in [0.05, 0.1, 0.2, 0.5, 1.0]:
        config = base.copy()
        config["name"] = f"{dataset}_temp_{temp}"
        config["contrastive_temp"] = temp
        ablation_configs.append(config)

    # Ablation 5: Commitment loss weight
    for commit_weight in [0.01, 0.05, 0.1, 0.2, 0.5]:
        config = base.copy()
        config["name"] = f"{dataset}_commit_{commit_weight}"
        config["lambda_commit_loss"] = commit_weight
        config["vq_commitment_weight"] = commit_weight
        ablation_configs.append(config)

    # Ablation 6: Single vs Dual codebook
    config = base.copy()
    config["name"] = f"{dataset}_single_codebook"
    config["codebook_size2"] = 0  # This would need model modification
    ablation_configs.append(config)

    return ablation_configs

def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for Contrastive VQ Graph")
    parser.add_argument("--mode", type=str, choices=["sota", "ablation", "both"],
                        default="sota", help="Run mode: sota, ablation, or both")
    parser.add_argument("--datasets", nargs="+",
                        default=["cora", "citeseer", "pubmed"],
                        help="Datasets to run experiments on")
    parser.add_argument("--output_dir", type=str, default="ablation_results",
                        help="Output directory for results")
    parser.add_argument("--device", type=int, default=0, help="GPU device")

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"experiments": [], "summary": {}}

    if args.mode in ["sota", "both"]:
        print("\n" + "="*60)
        print("Running SOTA Experiments")
        print("="*60)

        sota_configs = get_sota_configs()
        for dataset in args.datasets:
            if dataset in sota_configs:
                config = sota_configs[dataset]
                config["device"] = args.device
                success = run_experiment(config, str(output_dir))
                results["experiments"].append({
                    "name": config["name"],
                    "dataset": dataset,
                    "type": "sota",
                    "success": success
                })

    if args.mode in ["ablation", "both"]:
        print("\n" + "="*60)
        print("Running Ablation Studies")
        print("="*60)

        for dataset in args.datasets:
            ablation_configs = get_ablation_configs(dataset)
            for config in ablation_configs:
                config["device"] = args.device
                config["num_exp"] = 3  # Fewer runs for ablation
                success = run_experiment(config, str(output_dir))
                results["experiments"].append({
                    "name": config["name"],
                    "dataset": dataset,
                    "type": "ablation",
                    "success": success
                })

    # Save summary
    results["summary"]["total"] = len(results["experiments"])
    results["summary"]["successful"] = sum(1 for e in results["experiments"] if e["success"])
    results["summary"]["failed"] = results["summary"]["total"] - results["summary"]["successful"]

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    print(f"Total experiments: {results['summary']['total']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()