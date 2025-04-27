import pandas as pd
import yaml
import logging
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset, DataLoader
import torch
import os # Import os for path joining
import numpy as np # Import numpy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MixedDataDataset(Dataset):
    """PyTorch Dataset for mixed categorical and numerical data."""
    def __init__(self, dataframe: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str], target_col: str, weight_col: str):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.target_col = target_col
        self.weight_col = weight_col

        # Store data as numpy arrays for efficiency, convert categoricals to category type first if not already
        # We assume preprocessing (like mapping cats to ints, scaling nums) happens *before* creating the Dataset
        self.categorical_data = {}
        for col in self.categorical_cols:
            # Ensure the column exists and handle potential missing values if necessary (e.g., fillna or ensure preprocessing handles it)
            if col in dataframe:
                # Data should already be integer-encoded by the preprocessor
                self.categorical_data[col] = dataframe[col].values
            else:
                logging.warning(f"Categorical column '{col}' not found in DataFrame during Dataset creation.")

        self.numerical_data = {}
        for col in self.numerical_cols:
             if col in dataframe:
                 # Data should already be scaled by the preprocessor
                 self.numerical_data[col] = dataframe[col].astype(np.float32).values
             else:
                 logging.warning(f"Numerical column '{col}' not found in DataFrame during Dataset creation.")

        self.targets = dataframe[self.target_col].astype(np.float32).values
        self.weights = dataframe[self.weight_col].astype(np.float32).values

        self.n_samples = len(dataframe)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cat_features = {col: torch.tensor(self.categorical_data[col][idx], dtype=torch.long) for col in self.categorical_cols if col in self.categorical_data}
        num_features = {col: torch.tensor(self.numerical_data[col][idx], dtype=torch.float32) for col in self.numerical_cols if col in self.numerical_data}

        # Combine numerical features into a single tensor if needed by the model
        # Or keep them separate if the model handles dict input
        # For now, let's keep them separate, the model can concatenate

        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)

        # Return a dictionary for clarity
        return {
            'categorical': cat_features,
            'numerical': num_features,
            'target': target,
            'weight': weight
        }

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        raise

def load_data(config: Dict[str, Any], config_dir: str) -> pd.DataFrame:
    """Loads the dataset specified in the configuration."""
    data_path = config['data_path']
    # Handle relative paths in config: assume relative to config file's directory
    if not os.path.isabs(data_path):
        # Join and then normalize the path to correctly handle '../'
        data_path = os.path.normpath(os.path.join(config_dir, data_path))

    logging.info(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
         logging.error(f"Data file not found at {data_path}")
         raise FileNotFoundError(f"Data file not found at {data_path}")
    try:
        df = pd.read_parquet(data_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        # Basic validation
        required_cols = config['feature_cols'] + [config['target_col'], config['weight_col'], config['split_col']]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns in dataset: {missing_cols}")
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {data_path}: {e}")
        raise

def get_context_data(df: pd.DataFrame, context_name: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training and validation sets for a given context.

    Args:
        df: The full DataFrame.
        context_name: The name of the context (key in config['contexts']).
        config: The loaded configuration dictionary.

    Returns:
        A tuple containing (train_df, validation_df).
    """
    split_col = config['split_col']
    contexts = config['contexts']

    if context_name not in contexts:
        logging.error(f"Context '{context_name}' not defined in configuration.")
        raise ValueError(f"Context '{context_name}' not defined in configuration.")

    validation_values = contexts[context_name]['validation_values']
    logging.info(f"Splitting data for context '{context_name}'. Validation values in '{split_col}': {validation_values}")

    # Ensure validation values are of the same type as the split column for proper filtering
    try:
        if pd.api.types.is_numeric_dtype(df[split_col].dtype):
            validation_values = [type(df[split_col].iloc[0])(v) for v in validation_values]
        elif pd.api.types.is_string_dtype(df[split_col].dtype):
             validation_values = [str(v) for v in validation_values]
        # Add more type checks if necessary (e.g., for datetime)
    except Exception as e:
        logging.warning(f"Could not rigorously type-match validation values for column '{split_col}'. Proceeding with original types. Error: {e}")

    validation_mask = df[split_col].isin(validation_values)
    validation_df = df[validation_mask]
    train_df = df[~validation_mask]

    logging.info(f"Context '{context_name}': Train size={len(train_df)}, Validation size={len(validation_df)}")

    if len(validation_df) == 0:
        logging.warning(f"Context '{context_name}': No validation data found for values {validation_values} in column '{split_col}'. Check config and data.")
    if len(train_df) == 0:
         logging.warning(f"Context '{context_name}': No training data found (all data matched validation values). Check config and data.")

    return train_df, validation_df

def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for training and validation sets.
    Assumes dataframes contain preprocessed (scaled, encoded) data.
    """
    categorical_cols = config['categorical_cols']
    numerical_cols = [col for col in config['feature_cols'] if col not in categorical_cols]
    target_col = config['target_col']
    weight_col = config['weight_col']

    train_dataset = MixedDataDataset(
        dataframe=train_df,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_col=target_col,
        weight_col=weight_col
    )
    val_dataset = MixedDataDataset(
        dataframe=val_df,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        target_col=target_col,
        weight_col=weight_col
    )

    # Consider adding num_workers and pin_memory for performance if relevant
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logging.info(f"Created DataLoaders: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
    return train_loader, val_loader

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Assumes running from the root directory where preembedder/ folder exists
    # And that generate_data.py has been run
    CONFIG_PATH = './configs/sample_config.yaml' # Adjust path if needed
    CONFIG_DIR = os.path.dirname(CONFIG_PATH)

    try:
        cfg = load_config(CONFIG_PATH)
        full_dataframe = load_data(cfg, CONFIG_DIR)

        context = 'context_0' # Example context
        train_data, val_data = get_context_data(full_dataframe, context, cfg)

        print(f"\nData shapes for {context}: Train={train_data.shape}, Val={val_data.shape}")

        # ---- Preprocessing Placeholder ----
        # In a real pipeline, preprocessing would happen here before DataLoader creation
        # For this test, we use the raw data, assuming it matches expectations of MixedDataDataset
        print("\nNote: Bypassing preprocessing for this example. DataLoaders will use raw data.")
        # Example: Fit and transform scalers/encoders on train_data
        # Example: Apply transforms to val_data
        # For now, just assign them back (NO preprocessing)
        preprocessed_train_df = train_data
        preprocessed_val_df = val_data
        # ----------------------------------

        if not preprocessed_train_df.empty and not preprocessed_val_df.empty:
            batch_sz = cfg.get('batch_size', 256)
            train_dl, val_dl = create_dataloaders(preprocessed_train_df, preprocessed_val_df, cfg, batch_sz)

            # Test iterating through one batch
            print("\nTesting DataLoader output for one batch...")
            for batch in train_dl:
                print("Batch Keys:", batch.keys())
                print("Categorical Features Keys:", batch['categorical'].keys())
                # Example shape check for one categorical feature's batch
                if cfg['categorical_cols']:
                    first_cat_col = cfg['categorical_cols'][0]
                    print(f"Shape of '{first_cat_col}' batch: {batch['categorical'][first_cat_col].shape}")
                print("Numerical Features Keys:", batch['numerical'].keys())
                 # Example shape check for one numerical feature's batch
                num_cols = [col for col in cfg['feature_cols'] if col not in cfg['categorical_cols']]
                if num_cols:
                    first_num_col = num_cols[0]
                    print(f"Shape of '{first_num_col}' batch: {batch['numerical'][first_num_col].shape}")
                print("Target Shape:", batch['target'].shape)
                print("Weight Shape:", batch['weight'].shape)
                break # Only inspect the first batch
        else:
            print("Skipping DataLoader creation due to empty train or validation set.")

    except FileNotFoundError:
        print(f"Error: Ensure the config file exists at '{CONFIG_PATH}' and the data file specified within it exists.")
        print("Maybe run `python scripts/generate_data.py` first?")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
