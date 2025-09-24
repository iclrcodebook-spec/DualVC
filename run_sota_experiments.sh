#!/bin/bash

# Script to run SOTA experiments for Contrastive VQ Graph Model
# This script runs the best configurations for each dataset to reproduce paper results

echo "=========================================="
echo "Running SOTA Experiments for ICLR 2025"
echo "=========================================="

# Create results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="sota_results_${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

# Function to run experiment
run_experiment() {
    DATASET=$1
    echo ""
    echo "=========================================="
    echo "Running experiment on ${DATASET}"
    echo "=========================================="

    case ${DATASET} in
        "cora")
            python train_contrastive_vq.py \
                --dataset cora \
                --data_path ./data \
                --output_path ${RESULTS_DIR} \
                --num_exp 5 \
                --seed 0 \
                --device 0 \
                --num_gnn_layers 2 \
                --gnn_hidden_dim 512 \
                --gnn_output_dim 512 \
                --vq_dim 512 \
                --codebook_size1 512 \
                --codebook_size2 512 \
                --projection_head_hidden_dim 128 \
                --projection_head_output_dim 64 \
                --learning_rate 0.001 \
                --weight_decay 1e-5 \
                --max_epoch 200 \
                --patience 50 \
                --batch_size 256 \
                --contrastive_temp 0.2 \
                --lambda_commit_loss 0.1 \
                --vq_commitment_weight 0.1 \
                --vq_decay 0.99 \
                --dropout_ratio 0.1 \
                --norm_type batch \
                --activation_gnn relu \
                --vq_use_cosine_sim \
                --vq_kmeans_init \
                --num_linear_probe_runs 5 \
                --output_json ${RESULTS_DIR}/cora_results.json
            ;;

        "citeseer")
            python train_contrastive_vq.py \
                --dataset citeseer \
                --data_path ./data \
                --output_path ${RESULTS_DIR} \
                --num_exp 5 \
                --seed 0 \
                --device 0 \
                --num_gnn_layers 2 \
                --gnn_hidden_dim 512 \
                --gnn_output_dim 512 \
                --vq_dim 512 \
                --codebook_size1 512 \
                --codebook_size2 512 \
                --projection_head_hidden_dim 128 \
                --projection_head_output_dim 64 \
                --learning_rate 0.001 \
                --weight_decay 1e-5 \
                --max_epoch 200 \
                --patience 50 \
                --batch_size 256 \
                --contrastive_temp 0.2 \
                --lambda_commit_loss 0.1 \
                --vq_commitment_weight 0.1 \
                --vq_decay 0.99 \
                --dropout_ratio 0.1 \
                --norm_type batch \
                --activation_gnn relu \
                --vq_use_cosine_sim \
                --vq_kmeans_init \
                --num_linear_probe_runs 5 \
                --output_json ${RESULTS_DIR}/citeseer_results.json
            ;;

        "pubmed")
            python train_contrastive_vq.py \
                --dataset pubmed \
                --data_path ./data \
                --output_path ${RESULTS_DIR} \
                --num_exp 5 \
                --seed 0 \
                --device 0 \
                --num_gnn_layers 2 \
                --gnn_hidden_dim 512 \
                --gnn_output_dim 512 \
                --vq_dim 512 \
                --codebook_size1 512 \
                --codebook_size2 512 \
                --projection_head_hidden_dim 128 \
                --projection_head_output_dim 64 \
                --learning_rate 0.001 \
                --weight_decay 1e-5 \
                --max_epoch 200 \
                --patience 50 \
                --batch_size 512 \
                --contrastive_temp 0.2 \
                --lambda_commit_loss 0.1 \
                --vq_commitment_weight 0.1 \
                --vq_decay 0.99 \
                --dropout_ratio 0.1 \
                --norm_type batch \
                --activation_gnn relu \
                --vq_use_cosine_sim \
                --vq_kmeans_init \
                --num_linear_probe_runs 5 \
                --output_json ${RESULTS_DIR}/pubmed_results.json
            ;;

        "a-computer")
            python train_contrastive_vq.py \
                --dataset a-computer \
                --data_path ./data \
                --output_path ${RESULTS_DIR} \
                --num_exp 5 \
                --seed 0 \
                --device 0 \
                --num_gnn_layers 2 \
                --gnn_hidden_dim 512 \
                --gnn_output_dim 512 \
                --vq_dim 512 \
                --codebook_size1 512 \
                --codebook_size2 512 \
                --projection_head_hidden_dim 128 \
                --projection_head_output_dim 64 \
                --learning_rate 0.001 \
                --weight_decay 1e-5 \
                --max_epoch 200 \
                --patience 50 \
                --batch_size 512 \
                --contrastive_temp 0.2 \
                --lambda_commit_loss 0.1 \
                --vq_commitment_weight 0.1 \
                --vq_decay 0.99 \
                --dropout_ratio 0.1 \
                --norm_type batch \
                --activation_gnn relu \
                --vq_use_cosine_sim \
                --vq_kmeans_init \
                --num_linear_probe_runs 5 \
                --output_json ${RESULTS_DIR}/a-computer_results.json
            ;;

        "a-photo")
            python train_contrastive_vq.py \
                --dataset a-photo \
                --data_path ./data \
                --output_path ${RESULTS_DIR} \
                --num_exp 5 \
                --seed 0 \
                --device 0 \
                --num_gnn_layers 2 \
                --gnn_hidden_dim 512 \
                --gnn_output_dim 512 \
                --vq_dim 512 \
                --codebook_size1 512 \
                --codebook_size2 512 \
                --projection_head_hidden_dim 128 \
                --projection_head_output_dim 64 \
                --learning_rate 0.001 \
                --weight_decay 1e-5 \
                --max_epoch 200 \
                --patience 50 \
                --batch_size 512 \
                --contrastive_temp 0.2 \
                --lambda_commit_loss 0.1 \
                --vq_commitment_weight 0.1 \
                --vq_decay 0.99 \
                --dropout_ratio 0.1 \
                --norm_type batch \
                --activation_gnn relu \
                --vq_use_cosine_sim \
                --vq_kmeans_init \
                --num_linear_probe_runs 5 \
                --output_json ${RESULTS_DIR}/a-photo_results.json
            ;;

        "ogbn-arxiv")
            python train_contrastive_vq.py \
                --dataset ogbn-arxiv \
                --data_path ./data \
                --output_path ${RESULTS_DIR} \
                --num_exp 3 \
                --seed 0 \
                --device 0 \
                --num_gnn_layers 3 \
                --gnn_hidden_dim 512 \
                --gnn_output_dim 512 \
                --vq_dim 512 \
                --codebook_size1 1024 \
                --codebook_size2 1024 \
                --projection_head_hidden_dim 256 \
                --projection_head_output_dim 128 \
                --learning_rate 0.001 \
                --weight_decay 1e-5 \
                --max_epoch 600 \
                --patience 50 \
                --batch_size 2560 \
                --contrastive_temp 0.2 \
                --lambda_commit_loss 0.1 \
                --vq_commitment_weight 0.1 \
                --vq_decay 0.99 \
                --dropout_ratio 0.1 \
                --norm_type batch \
                --activation_gnn relu \
                --vq_use_cosine_sim \
                --vq_kmeans_init \
                --num_linear_probe_runs 5 \
                --output_json ${RESULTS_DIR}/ogbn-arxiv_results.json
            ;;

        "ogbn-products")
            python train_contrastive_vq.py \
                --dataset ogbn-products \
                --data_path ./data \
                --output_path ${RESULTS_DIR} \
                --num_exp 3 \
                --seed 0 \
                --device 0 \
                --num_gnn_layers 3 \
                --gnn_hidden_dim 512 \
                --gnn_output_dim 512 \
                --vq_dim 512 \
                --codebook_size1 2048 \
                --codebook_size2 2048 \
                --projection_head_hidden_dim 256 \
                --projection_head_output_dim 128 \
                --learning_rate 0.001 \
                --weight_decay 1e-5 \
                --max_epoch 600 \
                --patience 50 \
                --batch_size 5120 \
                --contrastive_temp 0.2 \
                --lambda_commit_loss 0.1 \
                --vq_commitment_weight 0.1 \
                --vq_decay 0.99 \
                --dropout_ratio 0.1 \
                --norm_type batch \
                --activation_gnn relu \
                --vq_use_cosine_sim \
                --vq_kmeans_init \
                --num_linear_probe_runs 5 \
                --output_json ${RESULTS_DIR}/ogbn-products_results.json
            ;;

        *)
            echo "Unknown dataset: ${DATASET}"
            return 1
            ;;
    esac

    if [ $? -eq 0 ]; then
        echo "✓ ${DATASET} experiment completed successfully"
    else
        echo "✗ ${DATASET} experiment failed"
    fi
}

# Run experiments based on command line arguments
if [ $# -eq 0 ]; then
    # Run all small datasets by default
    DATASETS="cora citeseer pubmed a-computer a-photo"
else
    DATASETS="$@"
fi

echo "Will run experiments on: ${DATASETS}"
echo ""

for dataset in ${DATASETS}; do
    run_experiment ${dataset}
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "=========================================="

# Create summary
echo ""
echo "Creating summary of results..."
python -c "
import json
import os
import sys

results_dir = '${RESULTS_DIR}'
datasets = '${DATASETS}'.split()

print('\\nSummary of Results:')
print('-' * 50)

for dataset in datasets:
    result_file = os.path.join(results_dir, f'{dataset}_results.json')
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            if 'mean_accuracy' in data:
                print(f'{dataset:15s}: {data[\"mean_accuracy\"]:.4f} ± {data.get(\"std_accuracy\", 0.0):.4f}')
            else:
                print(f'{dataset:15s}: No accuracy data found')
    else:
        print(f'{dataset:15s}: Results file not found')

print('-' * 50)
"