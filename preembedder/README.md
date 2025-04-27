# PreEmbedder: Context-Aware Categorical Feature Embeddings

## Purpose

PreEmbedder is a Python package designed to train neural network models for the primary purpose of extracting meaningful, context-aware embeddings for categorical features. It uses a supervised learning task (defined by a target variable in the data) to guide the embedding learning process.

The key idea is to handle situations where the optimal representation of a categorical feature might change depending on the "context" (e.g., time period, geographical region, product category). The package trains separate models (or uses context-specific data splits) for each defined context, allowing the embeddings to adapt accordingly.

## Features

*   **Context-Specific Training:** Trains models considering different contexts defined in the data.
*   **Supervised Embedding Learning:** Uses a specified target variable and supervised training objective to learn embeddings.
*   **Flexible Preprocessing:** Handles numerical (scaling) and categorical (vocabulary creation, embedding) features.
*   **Hyperparameter Optimization (HPO):** Integrates with `hyperopt` to automatically find optimal hyperparameters (embedding dimensions, MLP architecture, learning rate, etc.) using Tree-structured Parzen Estimators (TPE).
*   **Configurable Workflow:** Uses YAML files for easy configuration of data paths, features, context definitions, HPO settings, and training parameters.
*   **Artifact Saving:** Saves trained embeddings, best hyperparameters, and preprocessing artifacts (scalers, vocabularies) for downstream use.

## Implementation Overview

The workflow executed by `main.py` consists of the following steps:

1.  **Configuration Loading:** Loads settings from the specified YAML file.
2.  **Hyperparameter Optimization (Optional):** If `hpo.enabled` is `true` in the config, `run_hpo` is called. This function uses `hyperopt` to explore the defined `SEARCH_SPACE` (`hpo.py`). For each trial, it trains models across all contexts (using `hpo_objective` and `train_epoch`/`validate_epoch` from `trainer.py`) for a limited number of epochs (`hpo.epochs`), potentially stopping early (`hpo.early_stopping_patience`). The objective is to maximize the mean validation weighted correlation across contexts.
3.  **Final Parameter Selection:** The best parameters found via HPO (or the defaults from the config if HPO is skipped) are selected for the final training run.
4.  **Data Loading & Splitting:** Loads the full dataset (`data_loader.py`) and splits it into training/validation sets for each context based on `context_col` and `split_col`.
5.  **Preprocessing:** For each context, numerical features are scaled, and vocabularies are built for categorical features (`preprocessing.py`).
6.  **Final Model Training:** For each context, a `PreEmbedderNet` model (`model.py`) is initialized with the final hyperparameters and context-specific vocabularies. The model is trained on the context's training data for the full number of `epochs` specified in the top-level config (`trainer.py`).
7.  **Embedding Extraction:** After training, the learned embeddings for categorical features are extracted from each context's model.
8.  **Result Saving:** The extracted embeddings, final hyperparameters, and preprocessing artifacts are saved to the specified `output_dir` (`utils.py`).

## Dependencies

The core dependencies are listed in `requirements.txt`:

*   `pandas`
*   `numpy`
*   `pyarrow` (for Parquet support)
*   `scikit-learn` (for preprocessing)
*   `torch` (for neural network models and training)
*   `hyperopt` (for hyperparameter optimization)
*   `pyyaml` (for configuration file parsing)

Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

## Configuration

The package uses a YAML configuration file to control its behavior. See the detailed documentation for all options:

[**Full Configuration Documentation**](../docs/CONFIGURATION.md)

Key sections include:

*   Data paths (`data_path`, `output_dir`)
*   Feature and context definitions (`feature_cols`, `categorical_cols`, `context_col`, `split_col`, `contexts`)
*   Model architecture (`embedding_dim`, `mlp_layers`, `mlp_dropout`)
*   Training parameters (`learning_rate`, `batch_size`, `epochs`)
*   HPO settings (`hpo` section: `enabled`, `num_trials`, `epochs`, `early_stopping_patience`)

## Usage

1.  **Prepare Data:** Ensure your data is accessible (e.g., as a Parquet or CSV file).
2.  **Create Configuration:** Create a YAML configuration file (e.g., `configs/my_config.yaml`) specifying paths, columns, contexts, and desired HPO/training settings. Refer to `configs/sample_config.yaml` and `docs/CONFIGURATION.md`.
3.  **Run the Workflow:** Execute the `main.py` script from the directory containing the `preembedder` package directory (i.e., the directory containing `main.py`, `hpo.py`, etc.), providing the path to your configuration file. Ensure your virtual environment is activated.

```bash
# Example: Assuming you are in the directory containing the preembedder package
# Activate your virtual environment (e.g., source ../.venv/bin/activate)

python -m preembedder.main --config ../configs/my_config.yaml
```

*Note:* The command uses `python -m preembedder.main` to correctly handle relative imports within the package structure.

4.  **Check Results:** Find the extracted embeddings (`context_embeddings.json`), best hyperparameters (`best_hyperparameters.json`), and preprocessing artifacts (`preprocessing_artifacts.pkl`) in the specified `output_dir`.
