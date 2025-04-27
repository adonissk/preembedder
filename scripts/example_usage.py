import pandas as pd
import json
import numpy as np
# Assume preprocessing functions/classes (like scalers, vocabularies)
# are available from the main tool or saved separately.
# from preembedder.preprocessing import load_preprocessing_artifacts

# Placeholder for loading preprocessing artifacts (scalers, vocabs)
def load_preprocessing_artifacts(path):
    print(f"Warning: Placeholder function load_preprocessing_artifacts called from {path}")
    # In reality, this would load saved scalers and vocabulary mappings.
    # For this example, we'll simulate having some basic info.
    # Returning dummy data for demonstration
    class DummyScaler:
        def transform(self, data):
            # Simple scaling simulation
            # Return numpy array to match sklearn scaler behavior
            return ((data - data.mean()) / (data.std() + 1e-6)).values

    return {
        'context_0': {
            'num_scalers': {'num_feature_1': DummyScaler(), 'num_feature_2': DummyScaler(), 'noise_feature': DummyScaler()},
            'cat_vocabs': {
                'cat_feature_str': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, '__UNKNOWN__': 0},
                'cat_feature_int': {100: 1, 101: 2, 102: 3, 103: 4, 104: 5, 105: 6, 106: 7, 107: 8, 108: 9, 109: 10, '__UNKNOWN__': 0}
            }
        },
        # ... add dummy artifacts for other contexts if needed for testing
    }

def transform_data_with_embeddings(
    data_path: str,
    embeddings_path: str,
    preprocessing_artifacts_path: str,
    target_context: str,
    feature_cols: list,
    categorical_cols: list,
    output_path: str = None
):
    """
    Loads data, applies preprocessing, replaces categorical features
    with their learned embeddings for a specific context.

    Args:
        data_path: Path to the input data (e.g., Parquet).
        embeddings_path: Path to the JSON file containing learned embeddings.
        preprocessing_artifacts_path: Path to saved preprocessing objects (scalers, vocabs).
        target_context: The specific context for which to apply embeddings.
        feature_cols: List of original feature columns used in training.
        categorical_cols: List of categorical columns to replace with embeddings.
        output_path: Optional path to save the transformed DataFrame.

    Returns:
        pandas.DataFrame: The transformed data with embeddings.
    """

    # 1. Load data
    df = pd.read_parquet(data_path)
    print(f"Loaded data: {df.shape}")

    # 2. Load Embeddings
    with open(embeddings_path, 'r') as f:
        all_embeddings = json.load(f)

    if target_context not in all_embeddings:
        raise ValueError(f"Context '{target_context}' not found in embeddings file.")
    context_embeddings = all_embeddings[target_context]
    print(f"Loaded embeddings for context: '{target_context}'")

    # Infer embedding dimension from the first embedding vector
    any_cat_col = list(context_embeddings.keys())[0]
    any_val = list(context_embeddings[any_cat_col].keys())[0]
    embedding_dim = len(context_embeddings[any_cat_col][any_val])
    print(f"Inferred embedding dimension: {embedding_dim}")

    # 3. Load Preprocessing Artifacts (Scalers, Vocabs)
    # This part is crucial and depends on how artifacts are saved by the main tool.
    all_artifacts = load_preprocessing_artifacts(preprocessing_artifacts_path)
    if target_context not in all_artifacts:
        raise ValueError(f"Preprocessing artifacts for context '{target_context}' not found.")
    context_artifacts = all_artifacts[target_context]
    num_scalers = context_artifacts['num_scalers']
    cat_vocabs = context_artifacts['cat_vocabs']

    # 4. Preprocess Data (using context-specific artifacts)
    transformed_features = {}
    numerical_cols = [col for col in feature_cols if col not in categorical_cols]

    # Scale numerical features
    for col in numerical_cols:
        if col in num_scalers:
            # Reshape needed for sklearn scalers
            scaled_data = num_scalers[col].transform(df[[col]])
            transformed_features[col] = scaled_data.flatten()
        else:
            print(f"Warning: No scaler found for numerical column '{col}' in context '{target_context}'. Using original data.")
            transformed_features[col] = df[col].values

    # Map categorical features to embeddings
    for col in categorical_cols:
        if col in context_embeddings and col in cat_vocabs:
            vocab = cat_vocabs[col]
            embeddings = context_embeddings[col]
            unknown_token = '__UNKNOWN__' # Must match the token used during training
            unknown_embedding = embeddings.get(unknown_token) # Get the learned unknown embedding

            if unknown_embedding is None:
                print(f"Warning: No '__UNKNOWN__' embedding found for '{col}'. Using zeros.")
                unknown_embedding = [0.0] * embedding_dim

            # Convert original values to string for lookup (important if using mixed types like int keys in JSON)
            original_values = df[col].astype(str).fillna("None") # Handle potential NaNs

            # Look up embeddings, use unknown embedding for unseen values
            # Note: The keys in `embeddings` must be strings if loaded from JSON
            embedded_vectors = np.array([
                embeddings.get(val, unknown_embedding) for val in original_values
            ])

            # Create new column names for the embedding dimensions
            for i in range(embedding_dim):
                transformed_features[f'{col}_emb_{i}'] = embedded_vectors[:, i]
        else:
            print(f"Warning: No embeddings or vocab found for categorical column '{col}' in context '{target_context}'. Skipping.")

    # 5. Combine into a new DataFrame
    df_transformed = pd.DataFrame(transformed_features)

    # Optionally add back other columns (target, weights, ids etc.) if needed
    if 'target' in df.columns:
        df_transformed['target'] = df['target']
    if 'weight' in df.columns:
         df_transformed['weight'] = df['weight']
    # Add any other columns you might need

    print(f"Transformed data shape: {df_transformed.shape}")
    print("Transformed data head:")
    print(df_transformed.head())

    # 6. Save output (optional)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_transformed.to_parquet(output_path, index=False)
        print(f"Transformed data saved to {output_path}")

    return df_transformed


if __name__ == "__main__":
    import os

    # --- Configuration (should match generate_data.py and sample_config.yaml) ---
    SYNTHETIC_DATA_PATH = "../data/synthetic_data.parquet"
    EMBEDDINGS_PATH = "../results/context_embeddings.json" # Assumed output from main tool
    PREPROCESSING_ARTIFACTS_PATH = "../results/preprocessing_artifacts.pkl" # Assumed output
    OUTPUT_TRANSFORMED_PATH = "../results/transformed_data_context_0.parquet"

    TARGET_CONTEXT = "context_0" # Example: transform data for context 0

    # These lists should ideally be loaded from the config used for training
    FEATURE_COLS = [
        'cat_feature_str',
        'cat_feature_int',
        'num_feature_1',
        'num_feature_2',
        'noise_feature'
    ]
    CATEGORICAL_COLS = [
        'cat_feature_str',
        'cat_feature_int'
    ]

    # --- Create Dummy Files for Demonstration ---
    # Ensure data directory exists (if generate_data.py wasn't run)
    if not os.path.exists(os.path.dirname(SYNTHETIC_DATA_PATH)):
         os.makedirs(os.path.dirname(SYNTHETIC_DATA_PATH))
    # Create dummy data if it doesn't exist
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        print(f"Warning: {SYNTHETIC_DATA_PATH} not found. Creating dummy data.")
        dummy_df = pd.DataFrame({
            'cat_feature_str': ['A', 'B', 'A', 'C', None, 'B'],
            'cat_feature_int': [101, 102, 101, 103, 104, 105],
            'num_feature_1': np.random.rand(6) * 10,
            'num_feature_2': np.random.rand(6) * 5,
            'noise_feature': np.random.rand(6),
            'target': np.random.rand(6),
            'weight': np.random.rand(6) + 0.1,
            'split_col': [0, 1, 0, 1, 0, 2] # Example split values
        })
        dummy_df.to_parquet(SYNTHETIC_DATA_PATH)

    # Create dummy embeddings file
    if not os.path.exists(os.path.dirname(EMBEDDINGS_PATH)):
         os.makedirs(os.path.dirname(EMBEDDINGS_PATH))
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Warning: {EMBEDDINGS_PATH} not found. Creating dummy embeddings.")
        dummy_embeddings = {
            'context_0': {
                'cat_feature_str': {
                    'A': [0.1, 0.2], 'B': [0.3, 0.4], 'C': [0.5, 0.6], 'D': [0.7, 0.8], 'E': [0.9, 1.0], 'F': [1.1, 1.2], 'G': [1.3, 1.4],
                    'None': [0.01, 0.02], # Embedding for explicit None/NaN mapped to "None"
                    '__UNKNOWN__': [0.0, 0.0] # Embedding for unseen values
                },
                'cat_feature_int': { # Keys must be strings for JSON
                    '100': [1.1, 1.2], '101': [1.3, 1.4], '102': [1.5, 1.6], '103': [1.7, 1.8], '104': [1.9, 2.0],
                    '105': [2.1, 2.2], '106': [2.3, 2.4], '107': [2.5, 2.6], '108': [2.7, 2.8], '109': [2.9, 3.0],
                     '__UNKNOWN__': [1.0, 1.0]
                }
            }
            # Add other contexts if needed
        }
        with open(EMBEDDINGS_PATH, 'w') as f:
            json.dump(dummy_embeddings, f, indent=2)

    # Create dummy preprocessing artifacts path (needed by placeholder function)
    if not os.path.exists(os.path.dirname(PREPROCESSING_ARTIFACTS_PATH)):
        os.makedirs(os.path.dirname(PREPROCESSING_ARTIFACTS_PATH))
    # No need to write a file for the placeholder load function

    # --- Run the Transformation ---
    try:
        df_new = transform_data_with_embeddings(
            data_path=SYNTHETIC_DATA_PATH,
            embeddings_path=EMBEDDINGS_PATH,
            preprocessing_artifacts_path=PREPROCESSING_ARTIFACTS_PATH, # Path passed to loader
            target_context=TARGET_CONTEXT,
            feature_cols=FEATURE_COLS,
            categorical_cols=CATEGORICAL_COLS,
            output_path=OUTPUT_TRANSFORMED_PATH
        )
    except Exception as e:
        print(f"\nError during transformation: {e}")
        print("Please ensure the synthetic data and dummy embeddings exist.")
        print("Run `python scripts/generate_data.py` if needed.")
