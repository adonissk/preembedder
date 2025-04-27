import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UNKNOWN_TOKEN = "__UNKNOWN__"
NAN_REPLACEMENT = "__NAN__" # String representation for NaN/None values in categorical features

def create_numerical_scalers(df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, StandardScaler]:
    """Fits StandardScaler for each numerical column.

    Args:
        df: The training DataFrame.
        numerical_cols: List of numerical column names.

    Returns:
        A dictionary mapping column name to fitted StandardScaler instance.
    """
    scalers = {}
    for col in numerical_cols:
        if col in df:
            scaler = StandardScaler()
            # Reshape because scaler expects 2D array
            data_to_fit = df[col].values.reshape(-1, 1)
            # Filter out NaNs before fitting
            nan_mask = ~np.isnan(data_to_fit.flatten())
            if np.any(nan_mask):
                scaler.fit(data_to_fit[nan_mask])
                scalers[col] = scaler
                logging.debug(f"Fitted StandardScaler for column '{col}'. Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")
            else:
                logging.warning(f"Column '{col}' contains only NaN values. Cannot fit scaler.")
        else:
            logging.warning(f"Numerical column '{col}' not found in DataFrame during scaler fitting.")
    return scalers

def create_categorical_vocabs(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Dict[Any, int]]:
    """Creates vocabulary mapping (value to int index) for each categorical column.

    Index 0 is reserved for UNKNOWN_TOKEN.
    NaN/None values are explicitly mapped to NAN_REPLACEMENT before creating vocab.

    Args:
        df: The training DataFrame.
        categorical_cols: List of categorical column names.

    Returns:
        A dictionary mapping column name to its vocabulary dictionary.
    """
    vocabs = {}
    for col in categorical_cols:
        if col in df:
            # Replace NaN/None with a consistent string representation
            unique_values = df[col].fillna(NAN_REPLACEMENT).unique()
            # Create vocab, starting index from 1 (0 is reserved)
            vocab = {value: i + 1 for i, value in enumerate(unique_values)}
            # Add the unknown token mapping
            vocab[UNKNOWN_TOKEN] = 0
            vocabs[col] = vocab
            logging.debug(f"Created vocabulary for column '{col}'. Size: {len(vocab)}")
        else:
            logging.warning(f"Categorical column '{col}' not found in DataFrame during vocab creation.")
    return vocabs

def transform_numerical_features(df: pd.DataFrame, scalers: Dict[str, StandardScaler]) -> pd.DataFrame:
    """Applies fitted scalers to numerical columns.

    If a scaler is missing for a column, that column is returned unchanged with a warning.
    NaN values are preserved during transformation.

    Args:
        df: DataFrame to transform.
        scalers: Dictionary mapping column name to fitted StandardScaler.

    Returns:
        DataFrame with numerical columns scaled.
    """
    df_transformed = df.copy()
    numerical_cols = list(scalers.keys()) # Get columns from the scalers dict

    for col in numerical_cols:
        if col in df_transformed:
            if col in scalers:
                scaler = scalers[col]
                # Transform non-NaN values
                original_data = df_transformed[col].values.reshape(-1, 1)
                nan_mask = np.isnan(original_data.flatten())
                scaled_data = original_data.copy()
                if np.any(~nan_mask):
                    scaled_data[~nan_mask] = scaler.transform(original_data[~nan_mask])
                df_transformed[col] = scaled_data.flatten()
            else:
                 # This case should ideally not happen if preprocessing is consistent, but handle defensively.
                 logging.warning(f"Scaler for numerical column '{col}' not found during transformation. Column kept as is.")
        else:
            # This case implies the column was dropped or missing in the input df, which might be okay or an error depending on workflow.
            logging.warning(f"Numerical column '{col}' expected by scaler was not found in the DataFrame to transform.")

    return df_transformed

def transform_categorical_features(df: pd.DataFrame, vocabs: Dict[str, Dict[Any, int]]) -> pd.DataFrame:
    """Applies vocabulary mapping to categorical columns.

    Values not found in the vocab are mapped to the UNKNOWN_TOKEN's index (0).
    NaN/None values are mapped using NAN_REPLACEMENT before lookup.

    Args:
        df: DataFrame to transform.
        vocabs: Dictionary mapping column name to its vocabulary dictionary.

    Returns:
        DataFrame with categorical columns integer-encoded.
    """
    df_transformed = df.copy()
    categorical_cols = list(vocabs.keys())
    unknown_index = 0 # By convention from create_categorical_vocabs

    for col in categorical_cols:
        if col in df_transformed:
            if col in vocabs:
                vocab = vocabs[col]
                # Handle NaN/None consistently
                original_series = df_transformed[col].fillna(NAN_REPLACEMENT)
                # Map values to indices, using unknown_index for unseen values
                df_transformed[col] = original_series.apply(lambda x: vocab.get(x, unknown_index))
                # Ensure the output is integer type
                df_transformed[col] = df_transformed[col].astype(int)
            else:
                 logging.warning(f"Vocabulary for categorical column '{col}' not found during transformation. Column kept as is.")
        else:
            logging.warning(f"Categorical column '{col}' expected by vocabulary was not found in the DataFrame to transform.")

    return df_transformed

def save_preprocessing_artifacts(artifacts: Dict[str, Any], filepath: str):
    """Saves preprocessing artifacts (scalers, vocabs) to a file using pickle."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(artifacts, f)
        logging.info(f"Preprocessing artifacts saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving preprocessing artifacts to {filepath}: {e}")
        raise

def load_preprocessing_artifacts(filepath: str) -> Dict[str, Any]:
    """Loads preprocessing artifacts from a file."""
    if not os.path.exists(filepath):
        logging.error(f"Preprocessing artifacts file not found at {filepath}")
        raise FileNotFoundError(f"Preprocessing artifacts file not found at {filepath}")
    try:
        with open(filepath, 'rb') as f:
            artifacts = pickle.load(f)
        logging.info(f"Preprocessing artifacts loaded successfully from {filepath}")
        return artifacts
    except Exception as e:
        logging.error(f"Error loading preprocessing artifacts from {filepath}: {e}")
        raise

def preprocess_context_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fits preprocessing artifacts (scalers, vocabs) on training data
    and transforms both training and validation data for a single context.

    Args:
        train_df: Raw training DataFrame for the context.
        val_df: Raw validation DataFrame for the context.
        config: The loaded configuration dictionary.

    Returns:
        A tuple containing:
        - preprocessed_train_df: Transformed training DataFrame.
        - preprocessed_val_df: Transformed validation DataFrame.
        - artifacts: Dictionary containing fitted scalers and vocabs.
    """
    categorical_cols = config['categorical_cols']
    numerical_cols = [col for col in config['feature_cols'] if col not in categorical_cols]

    logging.info(f"Starting preprocessing. Train shape: {train_df.shape}, Val shape: {val_df.shape}")

    # Fit on training data
    logging.info("Fitting numerical scalers...")
    scalers = create_numerical_scalers(train_df, numerical_cols)
    logging.info("Creating categorical vocabularies...")
    vocabs = create_categorical_vocabs(train_df, categorical_cols)

    artifacts = {'scalers': scalers, 'vocabs': vocabs}

    # Transform both train and validation data
    logging.info("Transforming training data...")
    preprocessed_train_df = transform_numerical_features(train_df, scalers)
    preprocessed_train_df = transform_categorical_features(preprocessed_train_df, vocabs)

    logging.info("Transforming validation data...")
    preprocessed_val_df = transform_numerical_features(val_df, scalers)
    preprocessed_val_df = transform_categorical_features(preprocessed_val_df, vocabs)

    # Add back columns not used as features (target, weight, split_col) if they exist
    # This is important for the DataLoader
    for col in [config['target_col'], config['weight_col'], config['split_col']]:
        if col in train_df:
            preprocessed_train_df[col] = train_df[col]
        if col in val_df:
             preprocessed_val_df[col] = val_df[col]

    logging.info(f"Preprocessing complete. Processed Train shape: {preprocessed_train_df.shape}, Processed Val shape: {preprocessed_val_df.shape}")

    # Sanity check: Ensure all expected feature columns are present after processing
    processed_feature_cols = numerical_cols + categorical_cols
    missing_train_cols = [col for col in processed_feature_cols if col not in preprocessed_train_df.columns]
    missing_val_cols = [col for col in processed_feature_cols if col not in preprocessed_val_df.columns]
    if missing_train_cols:
        logging.warning(f"Columns expected in features missing after preprocessing train data: {missing_train_cols}")
    if missing_val_cols:
        logging.warning(f"Columns expected in features missing after preprocessing validation data: {missing_val_cols}")


    return preprocessed_train_df, preprocessed_val_df, artifacts

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Assumes running from the root directory
    # Requires data_loader module and generated data
    from preembedder.data_loader import load_config, load_data, get_context_data

    CONFIG_PATH = './configs/sample_config.yaml'
    ARTIFACTS_OUTPUT_PATH = './results/test_preprocessing_artifacts.pkl'
    CONFIG_DIR = os.path.dirname(CONFIG_PATH)

    try:
        cfg = load_config(CONFIG_PATH)
        full_dataframe = load_data(cfg, CONFIG_DIR)

        context = 'context_1' # Use a different context for testing
        train_data_raw, val_data_raw = get_context_data(full_dataframe, context, cfg)

        print(f"\nRaw data shapes for {context}: Train={train_data_raw.shape}, Val={val_data_raw.shape}")

        if not train_data_raw.empty:
            # Perform preprocessing
            train_data_proc, val_data_proc, fitted_artifacts = preprocess_context_data(
                train_data_raw, val_data_raw, cfg
            )

            print("\nPreprocessing finished.")
            print(f"Processed Train Head:\n{train_data_proc.head()}")
            print(f"Processed Val Head:\n{val_data_proc.head()}")

            # Example: Check a categorical column mapping in validation
            cat_col_example = cfg['categorical_cols'][0]
            print(f"\nExample transformation for '{cat_col_example}':")
            print("Raw Validation Values (first 5, with NaN handling):")
            print(val_data_raw[cat_col_example].fillna(NAN_REPLACEMENT).head())
            print("Processed Validation Indices (first 5):")
            print(val_data_proc[cat_col_example].head())

            # Check a numerical column scaling
            num_col_example = [col for col in cfg['feature_cols'] if col not in cfg['categorical_cols']][0]
            print(f"\nExample transformation for '{num_col_example}':")
            print(f"Raw Validation Values (first 5): {val_data_raw[num_col_example].head().tolist()}")
            print(f"Processed Validation Values (first 5): {val_data_proc[num_col_example].head().tolist()}")
            print(f"Scaler Mean: {fitted_artifacts['scalers'][num_col_example].mean_[0]:.4f}, Scale: {fitted_artifacts['scalers'][num_col_example].scale_[0]:.4f}")

            # Save and load artifacts
            print(f"\nSaving artifacts to {ARTIFACTS_OUTPUT_PATH}...")
            save_preprocessing_artifacts(fitted_artifacts, ARTIFACTS_OUTPUT_PATH)
            print("Loading artifacts back...")
            loaded_artifacts = load_preprocessing_artifacts(ARTIFACTS_OUTPUT_PATH)
            print("Artifacts loaded successfully.")
            # Verify loaded artifacts (e.g., check keys, types)
            print(f"Loaded scaler keys: {list(loaded_artifacts['scalers'].keys())}")
            print(f"Loaded vocab keys: {list(loaded_artifacts['vocabs'].keys())}")

        else:
            print("Skipping preprocessing example due to empty raw training data.")

    except FileNotFoundError:
        print(f"Error: Ensure config/data exist. Maybe run `python scripts/generate_data.py`?")
    except ImportError:
        print("Error: Could not import from preembedder.data_loader. Ensure it exists and Python path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
