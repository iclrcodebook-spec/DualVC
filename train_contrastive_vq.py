import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
import dgl
import time
from datetime import datetime
from dataloader import load_data, OGB_data, CPF_data
from utils import (
    get_logger,
    set_seed,
    check_writable,
)
from contrastive_vq_model import DualVQGNN
from contrastive_loss import ContrastiveLoss
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def get_args():
    parser = argparse.ArgumentParser(description="Contrastive VQGraph Training")
    parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    parser.add_argument("--log_level", type=int, default=20)
    parser.add_argument("--console_log", action="store_true", default=True)
    parser.add_argument("--output_path", type=str, default="outputs_contrastive_vq")
    parser.add_argument("--num_exp", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_results", action="store_true", default=True)

    # Dataset
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset (cora, citeseer, pubmed, a-computer, a-photo, ogbn-arxiv, ogbn-products)")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument("--labelrate_train", type=int, default=20)
    parser.add_argument("--labelrate_val", type=int, default=30)
    parser.add_argument("--split_idx", type=int, default=0)

    # Model specific
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--gnn_hidden_dim", type=int, default=512) #256
    parser.add_argument("--gnn_output_dim", type=int, default=512, help="Output dim of GNN (input to VQ project_in)") #256
    parser.add_argument("--vq_dim", type=int, default=512, help="Dimension of codebook vectors (VectorQuantize codebook_dim)") #256
    parser.add_argument("--codebook_size1", type=int, default=512) #256
    parser.add_argument("--codebook_size2", type=int, default=512) #256
    parser.add_argument("--projection_head_hidden_dim", type=int, default=128) 
    parser.add_argument("--projection_head_output_dim", type=int, default=64)

    parser.add_argument("--vq_commitment_weight", type=float, default=0.1)
    parser.add_argument("--vq_decay", type=float, default=0.99)
    parser.add_argument("--vq_use_cosine_sim", action='store_true', default=True) # Ensure this is correctly set from CLI or default
    parser.add_argument("--vq_kmeans_init", action='store_true', default=True) # Ensure this is correctly set from CLI or default

    parser.add_argument("--dropout_ratio", type=float, default=0.1)
    parser.add_argument("--norm_type", type=str, default="batch", help="One of [none, batch, layer] for GNN")
    parser.add_argument("--activation_gnn", type=str, default="relu", help="Activation for GNN layers (relu, leaky_relu)")

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epoch", type=int, default=600)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2560, help="Node batch size for mini-batch training on large graphs")

    # Contrastive Loss
    parser.add_argument("--contrastive_temp", type=float, default=0.2)
    parser.add_argument("--lambda_commit_loss", type=float, default=0.1, help="Weight for VQ commitment losses in total loss") 

    # Add this line for the evaluation parameter:
    parser.add_argument("--num_linear_probe_runs", type=int, default=5, 
                        help="Number of runs for linear probe evaluation to get mean/std for a single contrastive model")
    
    parser.add_argument("--output_json", type=str, default=None, 
                        help="Output JSON file to save final results")
    
    args = parser.parse_args()
    return args

def save_results_to_json(output_json_path, results_data):
    """Save final results to JSON file for ablation study"""
    if output_json_path:
        try:
            import json
            with open(output_json_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Results saved to: {output_json_path}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

def train_contrastive_epoch(model, dataloader_or_graph_tuple, optimizer, contrastive_criterion, device, lambda_commit_loss):
    model.train()
    total_loss_epoch = 0
    total_contrast_loss_epoch = 0
    total_commit_loss_epoch = 0
    num_items_processed = 0

    is_dataloader = isinstance(dataloader_or_graph_tuple, dgl.dataloading.DataLoader) 

    if is_dataloader:
        for input_nodes, output_nodes, blocks in dataloader_or_graph_tuple:
            blocks = [blk.to(device) for blk in blocks]
            batch_feats = model.g_for_dataloader.ndata['feat'][input_nodes].to(device)
            
            projected1_full, projected2_full, commit_loss1, commit_loss2 = model(blocks, batch_feats) 

            map_input_nodes_to_idx = {nid.item(): i for i, nid in enumerate(input_nodes)}
            local_output_indices = [map_input_nodes_to_idx[nid.item()] for nid in output_nodes if nid.item() in map_input_nodes_to_idx]
            
            if not local_output_indices: continue

            projected1 = projected1_full[local_output_indices]
            projected2 = projected2_full[local_output_indices]

            if projected1.size(0) == 0: continue

            contrast_loss = contrastive_criterion(projected1, projected2)
            commit_loss_combined = commit_loss1 + commit_loss2
            
            loss = contrast_loss + lambda_commit_loss * commit_loss_combined

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item() * projected1.size(0)
            total_contrast_loss_epoch += contrast_loss.item() * projected1.size(0)
            total_commit_loss_epoch += commit_loss_combined.item() * projected1.size(0)
            num_items_processed += projected1.size(0)
        
        if num_items_processed == 0: return 0,0,0 
        return total_loss_epoch / num_items_processed, total_contrast_loss_epoch / num_items_processed, total_commit_loss_epoch / num_items_processed
    else: 
        g, feats = dataloader_or_graph_tuple
        g = g.to(device)
        feats = feats.to(device)
        projected1, projected2, commit_loss1, commit_loss2 = model(g, feats)
        if projected1.size(0) == 0: return 0,0,0
        contrast_loss = contrastive_criterion(projected1, projected2)
        commit_loss_combined = commit_loss1 + commit_loss2
        loss = contrast_loss + lambda_commit_loss * commit_loss_combined
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), contrast_loss.item(), commit_loss_combined.item()


def evaluate_node_classification(
    args_eval, # Pass the main args
    trained_contrastive_model, 
    graph_eval, # The DGL graph object
    features_eval, # Full feature matrix
    labels_eval, # Full labels tensor
    idx_train_eval, 
    idx_val_eval, # Validation set for the linear probe (optional, can be used for early stopping probe)
    idx_test_eval, 
    device_eval, 
    logger_eval,
    num_linear_probe_runs_eval # How many times to run the linear probe
):
    logger_eval.info("--- Starting Node Classification Evaluation ---")
    
    # Get GNN embeddings (pre-VQ)
    # Ensure graph and features are on the correct device for the model
    graph_eval = graph_eval.to(device_eval)
    features_eval = features_eval.to(device_eval)
    
    # Use the dedicated method from DualVQGNN
    node_embeddings = trained_contrastive_model.get_gnn_representations(graph_eval, features_eval)
    node_embeddings = node_embeddings.detach().cpu().numpy() # To NumPy for scikit-learn
    
    labels_np = labels_eval.cpu().numpy()

    # Prepare data splits for scikit-learn
    X_train = node_embeddings[idx_train_eval.cpu().numpy()]
    y_train = labels_np[idx_train_eval.cpu().numpy()]
    # X_val = node_embeddings[idx_val_eval.cpu().numpy()] # Optional for probe hyperparam tuning/early stopping
    # y_val = labels_np[idx_val_eval.cpu().numpy()]
    X_test = node_embeddings[idx_test_eval.cpu().numpy()]
    y_test = labels_np[idx_test_eval.cpu().numpy()]

    # Scale features (good practice for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    test_accuracies = []

    for run_idx in range(num_linear_probe_runs_eval):
        # Using scikit-learn's Logistic Regression as a common linear probe
        # Create a new classifier instance for each run to ensure independence
        clf = LogisticRegression(
            random_state=args_eval.seed + run_idx + 1000, # Vary seed for probe runs
            solver='liblinear', 
            multi_class='auto', 
            max_iter=300 # Increased max_iter
        )
        clf.fit(X_train_scaled, y_train)
        
        y_pred_test = clf.predict(X_test_scaled)
        acc_test = accuracy_score(y_test, y_pred_test)
        test_accuracies.append(acc_test)
        logger_eval.info(f"  Linear Probe Run {run_idx+1}/{num_linear_probe_runs_eval} | Test Accuracy: {acc_test:.4f}")

    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)
    
    logger_eval.info(f"Node Classification Evaluation Summary ({num_linear_probe_runs_eval} probe runs):")
    logger_eval.info(f"  Mean Test Accuracy: {mean_accuracy:.4f}")
    logger_eval.info(f"  Std Dev Test Accuracy: {std_accuracy:.4f}")
    logger_eval.info("--- Finished Node Classification Evaluation ---")
    
    return mean_accuracy, std_accuracy


def run(args, pre_loaded_data=None): # Renamed from previous run to main_run_for_one_seed or similar
    set_seed(args.seed) # Use the seed for this specific run
    # ... (device setup, output_dir, logger setup as before) ...
    
    if torch.cuda.is_available() and args.device >= 0: # Ensure device is set correctly
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    args.device = device

    # Dynamic output directory name based on more parameters
    output_dir_name_parts = [
        f"ds_{args.dataset}",
        f"cb_{args.codebook_size1}_{args.codebook_size2}",
        f"gnn_{args.num_gnn_layers}l_{args.gnn_hidden_dim}h_{args.gnn_output_dim}o",
        f"vqdim_{args.vq_dim}",
        f"proj_{args.projection_head_output_dim}",
        f"lr_{args.learning_rate}",
        f"temp_{args.contrastive_temp}",
        f"commitw_{args.lambda_commit_loss}"
    ]
    output_dir_name = "_".join(output_dir_name_parts)
    output_dir = Path.cwd().joinpath(args.output_path, output_dir_name, f"seed_{args.seed}")
    
    check_writable(output_dir, overwrite=True) # Usually False for multi-runs, but True if each seed gets own folder
    logger = get_logger(output_dir.joinpath("log_contrastive_train.txt"), args.console_log, args.log_level)
    logger.info(f"Output dir for this run: {output_dir}")
    logger.info(f"Current run args: {args}")

    if pre_loaded_data is not None:
        g, labels, idx_train, idx_val, idx_test = pre_loaded_data
        logger.info(f"Using pre-loaded dataset {args.dataset}")
        
        # Re-create splits with current seed for variety (optional)
        # Comment out this section if you want to use the same splits across seeds
        if args.dataset in ['cora', 'citeseer', 'pubmed','a-computer','a-photo','ogbn-arxiv','ogbn-products']:
            total_nodes = g.number_of_nodes()
            torch.manual_seed(args.seed)  # Set seed for reproducible splits
            indices = torch.randperm(total_nodes)
            train_size = int(0.6 * total_nodes)
            val_size = int(0.2 * total_nodes)
            
            idx_train = indices[:train_size]
            idx_val = indices[train_size:train_size + val_size]
            idx_test = indices[train_size + val_size:]
            logger.info(f"Re-created splits for seed {args.seed}: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
        
        logger.info(f"Using dataset {args.dataset} with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
    else:
        # Original data loading - THIS SHOULD NOT HAPPEN when running ablation
        logger.info("Loading data from file...")
        g, labels, idx_train, idx_val, idx_test = load_data(
            args.dataset, args.data_path, split_idx=args.split_idx, seed=args.seed,
            labelrate_train=args.labelrate_train, labelrate_val=args.labelrate_val,
        )
        logger.info(f"Loaded dataset {args.dataset} with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")

    
    feats = g.ndata['feat']
    args.feat_dim = feats.shape[1]
    
    model = DualVQGNN( # Use args for model parameters
        input_dim=args.feat_dim, gnn_hidden_dim=args.gnn_hidden_dim, output_dim_gnn=args.gnn_output_dim,
        num_gnn_layers=args.num_gnn_layers, codebook_size1=args.codebook_size1, codebook_size2=args.codebook_size2,
        vq_dim=args.vq_dim, activation_str=args.activation_gnn, norm_type=args.norm_type,
        dropout_ratio=args.dropout_ratio, decay=args.vq_decay, commitment_weight=args.vq_commitment_weight,
        use_cosine_sim=args.vq_use_cosine_sim, kmeans_init=args.vq_kmeans_init,
        projection_head_hidden_dim=args.projection_head_hidden_dim,
        projection_head_output_dim=args.projection_head_output_dim
    ).to(device)

    logger.info(f"Model: {model}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Data loading setup (same as your working version)
    is_large_graph = g.number_of_nodes() > 5000 or args.dataset in OGB_data
    train_dataloader_or_graph_tuple = None
    actual_contrastive_bs = 0
    if is_large_graph and args.batch_size > 0:
        logger.info(f"Using mini-batch dataloader with batch_size: {args.batch_size}")
        model.g_for_dataloader = g.clone().to(device) # Store graph for feature fetching, clone to be safe
        train_nid = torch.arange(g.number_of_nodes()).to(device)
        sampler = dgl.dataloading.NeighborSampler([5] * args.num_gnn_layers)
        train_dataloader_or_graph_tuple = dgl.dataloading.DataLoader(
            g.to(device), train_nid, sampler, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0
        )
        actual_contrastive_bs = args.batch_size
    else:
        logger.info("Using full-batch training.")
        actual_contrastive_bs = g.number_of_nodes()
        # For full batch, ensure features are on device if not already done by load_data
        train_dataloader_or_graph_tuple = (g.add_self_loop().to(device), feats.to(device))

    contrastive_criterion = ContrastiveLoss(initial_batch_size=actual_contrastive_bs, temperature=args.contrastive_temp, device=device).to(device)

    best_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, args.max_epoch + 1):
        epoch_start_time = time.time()
        avg_loss, avg_contrast_loss, avg_commit_loss = train_contrastive_epoch(
            model, train_dataloader_or_graph_tuple, optimizer,
            contrastive_criterion, device, args.lambda_commit_loss
        )
        epoch_duration = time.time() - epoch_start_time
        if epoch % args.eval_interval == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d} | Time: {epoch_duration:.2f}s | Loss: {avg_loss:.4f} | ContrastLoss: {avg_contrast_loss:.4f} | CommitLoss: {avg_commit_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            if args.save_results:
                torch.save(model.state_dict(), output_dir.joinpath("best_model.pth"))
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= args.patience and epoch > args.patience :
            logger.info(f"Early stopping triggered at epoch {epoch} after {args.patience} epochs of no improvement.")
            break
    
    total_training_time = time.time() - start_time
    logger.info(f"Contrastive training finished. Total time: {total_training_time:.2f}s. Best loss: {best_loss:.4f}")
    
    # Load the best model for evaluation

    if Path(output_dir.joinpath("best_model.pth")).exists():
        logger.info("Loading best contrastive model for downstream evaluation.")
        model.load_state_dict(torch.load(output_dir.joinpath("best_model.pth"), map_location=device,weights_only=True))
    else:
        logger.warning("No best_model.pth found from contrastive training; using current model state for evaluation.")

    if args.save_results: # Save the model that will be used for evaluation
        torch.save(model.state_dict(), output_dir.joinpath("model_for_eval.pth"))
        cb1, cb2 = model.get_codebook_tensors()
        if cb1 is not None and cb2 is not None:
            np.savez(output_dir.joinpath("codebooks_for_eval.npz"), codebook1=cb1.cpu().numpy(), codebook2=cb2.cpu().numpy())
            logger.info("Saved model and codebooks used for evaluation.")

    # Perform node classification evaluation
    # Ensure original g, feats, labels, and splits are used here.
    # The evaluate_node_classification function handles moving g and feats to device.
    mean_acc, std_acc = evaluate_node_classification(
        args, model, g, feats, labels, 
        idx_train, idx_val, idx_test, 
        device, logger, args.num_linear_probe_runs
    )
    
    return mean_acc, std_acc # Return results of this run


def main():
    args = get_args()
    
    # Centralized logger for aggregating results from multiple experiments
    # Ensure the base output_path exists for this aggregate log
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    logger_aggregate = get_logger(Path(args.output_path).joinpath(f"AGGREGATE_RESULTS_{args.dataset}.txt"), 
                                  args.console_log, args.log_level)
    logger_aggregate.info(f"Starting {args.num_exp} experiment(s) with base settings: {vars(args)}")

    print(f"Pre-loading dataset {args.dataset}...")
    try:
        g, labels, idx_train, idx_val, idx_test = load_data(
            args.dataset, args.data_path, split_idx=args.split_idx, seed=0,  # Use seed=0 for data loading
            labelrate_train=args.labelrate_train, labelrate_val=args.labelrate_val,
        )
        print(f"Successfully pre-loaded {args.dataset}")
    except Exception as e:
        print(f"Error pre-loading data: {e}")
        return

    all_experiment_mean_accuracies = []

    for i in range(args.num_exp):
        current_seed_for_run = args.seed + i 
        
        # Create a copy of args to modify the seed for the current run
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = current_seed_for_run
        
        logger_aggregate.info(f"--- Experiment {i+1}/{args.num_exp} | Seed: {run_args.seed} ---")
        try:
            # The 'run' function now performs contrastive training AND its own multi-run linear probe eval
            mean_accuracy_for_this_seed, std_dev_for_this_seed_probes = run(run_args, pre_loaded_data=(g, labels, idx_train, idx_val, idx_test))
            
            all_experiment_mean_accuracies.append(mean_accuracy_for_this_seed)
            logger_aggregate.info(f"Result for Seed {run_args.seed}: Mean Accuracy = {mean_accuracy_for_this_seed:.4f} (Probe StdDev = {std_dev_for_this_seed_probes:.4f})")
        except Exception as e:
            logger_aggregate.error(f"Error in experiment {i+1} (Seed {run_args.seed}): {e}", exc_info=True)
            all_experiment_mean_accuracies.append(np.nan) # Record failure if needed

    if all_experiment_mean_accuracies:
        valid_accuracies = [acc for acc in all_experiment_mean_accuracies if not np.isnan(acc)]
        if valid_accuracies:
            final_overall_mean_accuracy = np.mean(valid_accuracies)
            final_overall_std_dev_accuracy = np.std(valid_accuracies)
            
            summary_msg = (
                f"\n--- Overall Summary ({len(valid_accuracies)} successful runs out of {args.num_exp}) ---\n"
                f"Dataset: {args.dataset}\n"
                f"Mean Test Accuracy across seeds: {final_overall_mean_accuracy:.4f}\n"
                f"Std Dev Test Accuracy across seeds: {final_overall_std_dev_accuracy:.4f}\n"
                f"Individual run accuracies: {['{:.4f}'.format(acc) for acc in valid_accuracies]}"
            )
            logger_aggregate.info(summary_msg)
            print(summary_msg)
            
            # NEW: Save results to JSON for ablation study
            if args.output_json:
                results_data = {
                    'dataset': args.dataset,
                    'mean_accuracy': final_overall_mean_accuracy,
                    'std_accuracy': final_overall_std_dev_accuracy,
                    'individual_accuracies': valid_accuracies,
                    'num_successful_runs': len(valid_accuracies),
                    'total_runs': args.num_exp,
                    'hyperparameters': {
                        'vq_commitment_weight': args.vq_commitment_weight,
                        'lambda_commit_loss': args.lambda_commit_loss,
                        'contrastive_temp': args.contrastive_temp,
                        'learning_rate': args.learning_rate,
                        'codebook_size1': args.codebook_size1,
                        'codebook_size2': args.codebook_size2,
                        'gnn_hidden_dim': args.gnn_hidden_dim,
                        'gnn_output_dim': args.gnn_output_dim,
                        'vq_dim': args.vq_dim,
                        'num_gnn_layers': args.num_gnn_layers,
                        'max_epoch': args.max_epoch,
                        'patience': args.patience,
                    },
                    'timestamp': datetime.now().isoformat()
                }
                save_results_to_json(args.output_json, results_data)
        else:
            logger_aggregate.info("No successful experiment runs to aggregate.")
            print("No successful experiment runs to aggregate.")
            
            # Save failure result to JSON
            if args.output_json:
                results_data = {
                    'dataset': args.dataset,
                    'mean_accuracy': 0.0,
                    'error': 'All runs failed',
                    'timestamp': datetime.now().isoformat()
                }
                save_results_to_json(args.output_json, results_data)
    else:
        logger_aggregate.info("No experiments were run or all failed.")
        print("No experiments were run or all failed.")



if __name__ == "__main__":
    main()