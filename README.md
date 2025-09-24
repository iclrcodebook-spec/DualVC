# Contrastive Learning with Vector Quantized Graph Representations

This repository contains the implementation for our ICLR 2025 submission.

## Abstract

We propose a novel contrastive learning framework that leverages dual vector quantization (VQ) modules for learning robust graph representations. Our approach combines graph neural networks with vector quantization to create discrete latent representations that capture structural patterns in graph data.

## Requirements

```bash
conda create -n contrastive_vq python=3.9
conda activate contrastive_vq
pip install -r requirements.txt
```

## Dataset Preparation

Please download the datasets and place them in the `data/` directory:

- **CPF datasets** (cora, citeseer, pubmed, a-computer, a-photo): Download `.npz` files and rename as specified:
  - `amazon_electronics_computers.npz` → `a-computer.npz`
  - `amazon_electronics_photo.npz` → `a-photo.npz`

- **OGB datasets** (ogbn-arxiv, ogbn-products): Will be automatically downloaded when running the code.

## Running Experiments

### Training the Contrastive VQ Model

To train the contrastive VQ model on a dataset:

```bash
python train_contrastive_vq.py \
    --dataset cora \
    --data_path ./data \
    --num_gnn_layers 2 \
    --gnn_hidden_dim 512 \
    --gnn_output_dim 512 \
    --vq_dim 512 \
    --codebook_size1 512 \
    --codebook_size2 512 \
    --learning_rate 0.001 \
    --max_epoch 200 \
    --device 0
```

### Key Parameters

- `--dataset`: Dataset to use (cora, citeseer, pubmed, a-computer, a-photo, ogbn-arxiv, ogbn-products)
- `--num_gnn_layers`: Number of GNN layers (default: 2)
- `--gnn_hidden_dim`: Hidden dimension for GNN (default: 512)
- `--codebook_size1`, `--codebook_size2`: Sizes of the two codebooks (default: 512)
- `--vq_dim`: Dimension of codebook vectors (default: 512)
- `--contrastive_temp`: Temperature for contrastive loss (default: 0.2)
- `--learning_rate`: Learning rate (default: 0.001)
- `--max_epoch`: Maximum number of training epochs (default: 200)

### Evaluation

The model automatically performs linear probe evaluation during training to assess representation quality. Results include mean accuracy and standard deviation across multiple probe runs.

## File Structure

```
├── contrastive_vq_model.py    # Main model architecture
├── contrastive_loss.py        # Contrastive loss implementation
├── vq.py                       # Vector quantization modules
├── models.py                   # GNN and MLP model definitions
├── dataloader.py               # Data loading utilities
├── data_preprocess.py          # Data preprocessing functions
├── train_contrastive_vq.py     # Main training script
├── utils.py                    # Utility functions
├── train.conf.yaml             # Training configuration
└── requirements.txt            # Python dependencies
```

## Model Architecture

Our model consists of:
1. A GNN encoder that processes graph-structured data
2. Dual vector quantization modules that discretize representations
3. Projection heads for contrastive learning
4. A contrastive loss that encourages learning of discriminative features

## Reproducibility

To ensure reproducibility:
- Set random seeds using the `--seed` parameter
- Use the provided configuration files
- Follow the exact dataset preparation steps

## Results

The model achieves competitive performance across various graph benchmarks. Detailed results and ablation studies are provided in our paper.

## Citation

If you use this code for your research, please cite our paper:

```
@inproceedings{anonymous2025contrastive,
  title={Contrastive Learning with Vector Quantized Graph Representations},
  author={Anonymous Authors},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```