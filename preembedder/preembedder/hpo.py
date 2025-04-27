import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import torch
import torch.optim as optim
import numpy as np
import logging
import os
import copy
from typing import Dict, Any, List
import time

# Import necessary components from other modules
from .data_loader import load_data, get_context_data, create_dataloaders, load_config
from .preprocessing import preprocess_context_data
from .model import PreEmbedderNet
from .trainer import train_epoch, validate_epoch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the hyperparameter search space based on config structure (can be customized)
# We'll use the example structure from the sample_config.yaml comments
SEARCH_SPACE = {
    'embedding_dim': hp.choice('embedding_dim', [8, 16, 32, 64]),
    # Represent mlp_layers structure carefully for hyperopt
    'mlp_depth': hp.choice('mlp_depth', [1, 2, 3]), # Choose depth first
    'mlp_width': hp.choice('mlp_width', [32, 64, 128]), # Choose width
    # mlp_layers structure will be built based on depth and width in the objective
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
    'batch_size': hp.choice('batch_size', [128, 256, 512]),
    'mlp_dropout': hp.uniform('mlp_dropout', 0.0, 0.5)
}


def hpo_objective(params: Dict[str, Any], config: Dict[str, Any], config_dir: str, device: torch.device) -> Dict[str, Any]:
    """Objective function for hyperopt.

    Trains models for all contexts with the given parameters and returns the
    negative mean validation wcorr across contexts, along with other info.
    """
    trial_start_time = time.time()
    logging.info(f"--- Starting HPO Trial --- Parameters: {params}")

    # --- Setup based on params and config ---
    try:
        # Construct mlp_layers from params['mlp_depth'] and params['mlp_width']
        # Example: depth=2, width=64 -> [64, 64]
        mlp_layers = [params['mlp_width']] * params['mlp_depth']

        current_config = copy.deepcopy(config) # Avoid modifying the original config
        # Override config values with HPO params where applicable
        current_config['embedding_dim'] = params['embedding_dim']
        current_config['mlp_layers'] = mlp_layers
        current_config['mlp_dropout'] = params['mlp_dropout']
        current_config['learning_rate'] = params['learning_rate']
        current_config['batch_size'] = params['batch_size']

        # Load data once
        full_df = load_data(current_config, config_dir)
        context_names = list(current_config['contexts'].keys())
        num_contexts = len(context_names)

        models = {}
        optimizers = {}
        train_loaders = {}
        val_loaders = {}
        context_artifacts = {}
        all_context_vocabs = {}

        numerical_cols = [col for col in current_config['feature_cols'] if col not in current_config['categorical_cols']]
        categorical_cols = current_config['categorical_cols']

        # --- Per-Context Preparation ---
        for context_name in context_names:
            logging.debug(f"[HPO Trial] Preparing data for context: {context_name}")
            train_df_raw, val_df_raw = get_context_data(full_df, context_name, current_config)

            if train_df_raw.empty or val_df_raw.empty:
                logging.error(f"[HPO Trial] Context '{context_name}' has empty train or validation set. Skipping HPO trial.")
                # Return a high loss to discard this trial
                return {'loss': float('inf'), 'status': hyperopt.STATUS_FAIL, 'message': f"Empty data for {context_name}"}

            # Preprocess (fit on train, transform both)
            train_df_proc, val_df_proc, artifacts = preprocess_context_data(train_df_raw, val_df_raw, current_config)
            context_artifacts[context_name] = artifacts
            all_context_vocabs[context_name] = artifacts['vocabs'] # Store vocabs for model init

            # Create DataLoaders
            train_loader, val_loader = create_dataloaders(train_df_proc, val_df_proc, current_config, current_config['batch_size'])
            train_loaders[context_name] = train_loader
            val_loaders[context_name] = val_loader

            # Create Model (using context-specific vocabs but HPO hyperparams)
            model = PreEmbedderNet(
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                context_vocabs=all_context_vocabs[context_name], # Use the vocabs specific to this context
                embedding_dim=current_config['embedding_dim'],
                mlp_layers=current_config['mlp_layers'],
                mlp_dropout=current_config['mlp_dropout']
            ).to(device)
            models[context_name] = model

            # Create Optimizer
            optimizer = optim.Adam(model.parameters(), lr=current_config['learning_rate'])
            optimizers[context_name] = optimizer

        # --- Synchronized Training Loop with Early Stopping on Mean Metric ---
        # Read epochs from hpo config first, then top-level, then default
        num_epochs = current_config.get('hpo', {}).get('epochs', current_config.get('epochs', 10)) # Default 10 if not found
        patience = current_config.get('hpo', {}).get('early_stopping_patience', current_config.get('early_stopping_patience', 3)) # Default 3 if not found
        best_mean_val_wcorr = -float('inf') # We want to maximize wcorr
        epochs_no_improve = 0
        best_epoch = -1

        all_histories = {ctx: {'train_loss': [], 'val_loss': [], 'val_wcorr': []} for ctx in context_names}
        mean_wcorr_history = []

        logging.info(f"[HPO Trial] Starting synchronized training for {num_contexts} contexts...")
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            current_epoch_val_wcorrs = []

            # Train and validate each context model for one epoch
            for context_name in context_names:
                model = models[context_name]
                optimizer = optimizers[context_name]
                train_loader = train_loaders[context_name]
                val_loader = val_loaders[context_name]

                train_loss = train_epoch(model, train_loader, optimizer, device)
                val_loss, val_wcorr = validate_epoch(model, val_loader, device)

                all_histories[context_name]['train_loss'].append(train_loss)
                all_histories[context_name]['val_loss'].append(val_loss)
                all_histories[context_name]['val_wcorr'].append(val_wcorr)

                # Handle potential NaN wcorr - treat as worst score (-1)
                current_epoch_val_wcorrs.append(val_wcorr if not np.isnan(val_wcorr) else -1.0)

            # Calculate mean validation wcorr across contexts
            mean_val_wcorr = np.mean(current_epoch_val_wcorrs)
            mean_wcorr_history.append(mean_val_wcorr)

            epoch_duration = time.time() - epoch_start_time
            logging.info(f"[HPO Trial] Epoch {epoch + 1}/{num_epochs} | Mean Val wCorr: {mean_val_wcorr:.5f} | Time: {epoch_duration:.2f}s")

            # Early Stopping Check (based on mean wcorr)
            if mean_val_wcorr > best_mean_val_wcorr + 1e-5: # Add tolerance for floating point
                best_mean_val_wcorr = mean_val_wcorr
                epochs_no_improve = 0
                best_epoch = epoch + 1
                # Optional: Save best model state here if needed, but not required for HPO itself
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logging.info(f"[HPO Trial] Early stopping triggered at epoch {epoch + 1} due to no improvement in mean validation wcorr for {patience} epochs.")
                break

        trial_duration = time.time() - trial_start_time
        logging.info(f"[HPO Trial] Finished. Best Mean Val wCorr: {best_mean_val_wcorr:.5f} at epoch {best_epoch}. Duration: {trial_duration:.2f}s")

        # Hyperopt minimizes the 'loss' key
        return {
            'loss': -best_mean_val_wcorr, # Minimize negative correlation
            'status': STATUS_OK,
            'best_mean_wcorr': best_mean_val_wcorr,
            'best_epoch': best_epoch,
            'params': params, # Store params tried
            'trial_duration_s': trial_duration
        }

    except Exception as e:
        logging.error(f"[HPO Trial] Error during trial: {e}", exc_info=True)
        # Return high loss on failure
        return {'loss': float('inf'), 'status': hyperopt.STATUS_FAIL, 'message': str(e)}

def run_hpo(config: Dict[str, Any], config_dir: str) -> Dict[str, Any]:
    """Runs the hyperparameter optimization process.

    Args:
        config: The main configuration dictionary.
        config_dir: The directory containing the config file (for relative paths).

    Returns:
        The best hyperparameter set found.
    """
    if not config.get('hpo', {}).get('enabled', False):
        logging.info("HPO is disabled in the configuration. Skipping.")
        # Return default parameters from config if HPO is off
        # Construct mlp_layers from config
        mlp_layers = config.get('mlp_layers', [64, 32])
        mlp_depth = len(mlp_layers)
        mlp_width = mlp_layers[0] if mlp_layers else 32 # Simple guess if empty

        return {
            'embedding_dim': config.get('embedding_dim', 16),
            'mlp_depth': mlp_depth,
            'mlp_width': mlp_width,
            'learning_rate': config.get('learning_rate', 0.001),
            'batch_size': config.get('batch_size', 256),
            'mlp_dropout': config.get('mlp_dropout', 0.1)
        }

    hpo_config = config['hpo']
    num_trials = hpo_config.get('num_trials', 50)
    random_seed = config.get('random_seed', 42)

    # --- Define HPO Space --- (Could be made more dynamic based on config later)
    search_space = SEARCH_SPACE
    logging.info(f"Starting HPO with {num_trials} trials. Search Space: {search_space}")

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Run Optimization ---
    trials = Trials()
    # Use a lambda to pass fixed arguments (config, config_dir, device) to the objective
    objective_with_args = lambda params: hpo_objective(params, config, config_dir, device)

    best = fmin(
        fn=objective_with_args,
        space=search_space,
        algo=tpe.suggest, # Tree-structured Parzen Estimator
        max_evals=num_trials,
        trials=trials,
        rstate=np.random.default_rng(random_seed) # For reproducibility
    )

    # --- Get Best Parameters --- #
    # `best` returned by fmin has indices for choice selections.
    # `hyperopt.space_eval` converts these indices back to the actual values.
    best_params = hyperopt.space_eval(search_space, best)
    logging.info(f"HPO finished. Best parameters found: {best_params}")

    # You can also access detailed trial results from the `trials` object:
    # logging.debug(f"Best trial info: {trials.best_trial}")

    return best_params

# Example usage (optional, for testing - assumes config and data exist)
if __name__ == '__main__':
    print("HPO module example structure (requires full pipeline setup to run):")
    CONFIG_PATH = './configs/sample_config.yaml'
    CONFIG_DIR = os.path.dirname(CONFIG_PATH)

    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found at {CONFIG_PATH}. Cannot run HPO example.")
    else:
        try:
            cfg = load_config(CONFIG_PATH)
            # Make sure HPO is enabled in the config for testing
            cfg['hpo'] = cfg.get('hpo', {})
            cfg['hpo']['enabled'] = True
            cfg['hpo']['num_trials'] = 5 # Reduce trials for quick test
            cfg['epochs'] = 10 # Reduce epochs for quick test
            cfg['early_stopping_patience'] = 3

            print(f"Running HPO example with {cfg['hpo']['num_trials']} trials...")
            best_hyperparameters = run_hpo(cfg, CONFIG_DIR)

            print("\n--- HPO Example Finished ---")
            print("Best hyperparameters found:")
            print(best_hyperparameters)

        except FileNotFoundError as e:
             print(f"Error: Missing data or config file: {e}")
             print("Ensure synthetic data exists (`python scripts/generate_data.py`) and config paths are correct.")
        except Exception as e:
            print(f"An error occurred during HPO example: {e}")
            import traceback
            traceback.print_exc()
