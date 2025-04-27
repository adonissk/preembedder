import argparse
import os
import logging
import torch
import torch.optim as optim
import numpy as np
import random
import copy
import time

# Import project modules
from .data_loader import load_config, load_data, get_context_data, create_dataloaders
from .preprocessing import preprocess_context_data, save_preprocessing_artifacts
from .model import PreEmbedderNet
from .hpo import run_hpo
from .trainer import train_context
from .utils import save_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed: int):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def main():
    parser = argparse.ArgumentParser(description="PreEmbedder: Train models to extract categorical embeddings.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    start_time = time.time()
    logging.info("--- Starting PreEmbedder Workflow ---")

    # --- Load Configuration and Setup ---
    config_path = args.config
    config_dir = os.path.dirname(config_path)
    config = load_config(config_path)

    # Set seed for reproducibility
    set_seed(config.get('random_seed', 42))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Hyperparameter Optimization (Optional) ---
    logging.info("--- Step 1: Hyperparameter Optimization (if enabled) ---")
    best_hpo_params = run_hpo(config, config_dir)

    # Update the main config with the best HPO parameters or defaults
    final_params = copy.deepcopy(config) # Start with original config
    # Construct mlp_layers from best HPO params
    mlp_layers = [best_hpo_params['mlp_width']] * best_hpo_params['mlp_depth']
    final_params['embedding_dim'] = best_hpo_params['embedding_dim']
    final_params['mlp_layers'] = mlp_layers
    final_params['mlp_dropout'] = best_hpo_params['mlp_dropout']
    final_params['learning_rate'] = best_hpo_params['learning_rate']
    final_params['batch_size'] = best_hpo_params['batch_size']
    # Note: 'epochs' and 'early_stopping_patience' from original config are used for final training

    logging.info(f"Final training parameters: {final_params}")

    # Save the chosen hyperparameters
    output_dir = final_params.get('output_dir', './results')
    hyperparams_filename = final_params.get('hyperparams_filename', 'best_hyperparameters.json')
    hyperparams_path = os.path.join(output_dir, hyperparams_filename)
    # Ensure output dir exists relative to the *config* directory if path is relative
    if not os.path.isabs(output_dir):
        output_dir_abs = os.path.join(config_dir, output_dir)
    else:
        output_dir_abs = output_dir
    hyperparams_path_abs = os.path.join(output_dir_abs, hyperparams_filename)

    save_json(best_hpo_params, hyperparams_path_abs)
    logging.info(f"Best hyperparameters saved to {hyperparams_path_abs}")

    # --- Load Data ---
    logging.info("--- Step 2: Loading Full Dataset ---")
    full_df = load_data(final_params, config_dir)
    context_names = list(final_params['contexts'].keys())

    # --- Final Training and Embedding Extraction per Context ---
    logging.info("--- Step 3: Final Training and Embedding Extraction per Context ---")
    all_context_embeddings = {}
    all_context_artifacts = {}
    numerical_cols = [col for col in final_params['feature_cols'] if col not in final_params['categorical_cols']]
    categorical_cols = final_params['categorical_cols']

    for context_name in context_names:
        context_start_time = time.time()
        logging.info(f"--- Processing Context: {context_name} ---")

        # 1. Get context data split
        train_df_raw, val_df_raw = get_context_data(full_df, context_name, final_params)

        if train_df_raw.empty or val_df_raw.empty:
            logging.warning(f"Context '{context_name}' has empty train or validation set. Skipping final training for this context.")
            continue

        # 2. Preprocess data for this context
        logging.info(f"[{context_name}] Preprocessing data...")
        train_df_proc, val_df_proc, artifacts = preprocess_context_data(train_df_raw, val_df_raw, final_params)
        all_context_artifacts[context_name] = artifacts # Store artifacts

        # 3. Create DataLoaders
        train_loader, val_loader = create_dataloaders(train_df_proc, val_df_proc, final_params, final_params['batch_size'])

        # 4. Initialize Model and Optimizer with final parameters
        logging.info(f"[{context_name}] Initializing model...")
        model = PreEmbedderNet(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            context_vocabs=artifacts['vocabs'], # Use context-specific vocabs
            embedding_dim=final_params['embedding_dim'],
            mlp_layers=final_params['mlp_layers'],
            mlp_dropout=final_params['mlp_dropout']
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=final_params['learning_rate'])

        # 5. Train the model for this context
        # Note: We train for the full number of epochs specified in the config for the final run.
        # Alternatively, could use best_epoch from HPO if saved, or implement separate early stopping.
        logging.info(f"[{context_name}] Starting final training...")
        # TODO: Consider if final training should use train+val data or just train.
        # Using just train for now, as validation set was used for HPO selection.
        train_history = train_context(
            context_name=context_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, # Still use val_loader for monitoring during final training
            optimizer=optimizer,
            num_epochs=final_params['epochs'],
            device=device
        )
        # We don't strictly need the history here, but it might be useful for logging/debugging.
        final_val_wcorr = train_history['val_wcorr'][-1] if train_history['val_wcorr'] else float('nan')
        logging.info(f"[{context_name}] Final training complete. Final validation wCorr: {final_val_wcorr:.4f}")

        # 6. Extract Embeddings
        logging.info(f"[{context_name}] Extracting embeddings...")
        extracted_embeddings = model.extract_embeddings(device=torch.device("cpu")) # Extract on CPU
        all_context_embeddings[context_name] = extracted_embeddings

        context_duration = time.time() - context_start_time
        logging.info(f"--- Finished Context: {context_name} | Duration: {context_duration:.2f}s ---")

    # --- Save Results ---
    logging.info("--- Step 4: Saving Final Results ---")

    # Save combined embeddings
    embedding_filename = final_params.get('embedding_filename', 'context_embeddings.json')
    embedding_path_abs = os.path.join(output_dir_abs, embedding_filename)
    save_json(all_context_embeddings, embedding_path_abs)
    logging.info(f"Final embeddings saved to {embedding_path_abs}")

    # Save combined preprocessing artifacts (optional, but useful for downstream use)
    artifacts_filename = "preprocessing_artifacts.pkl" # Define a standard name
    artifacts_path_abs = os.path.join(output_dir_abs, artifacts_filename)
    # Structure artifacts for saving: {context_name: {'scalers': ..., 'vocabs': ...}}
    save_preprocessing_artifacts(all_context_artifacts, artifacts_path_abs)
    logging.info(f"Preprocessing artifacts saved to {artifacts_path_abs}")

    total_duration = time.time() - start_time
    logging.info(f"--- PreEmbedder Workflow Completed --- Total Duration: {total_duration:.2f}s ---")

if __name__ == "__main__":
    main() 