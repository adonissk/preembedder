import pandas as pd
import numpy as np
import random

def generate_synthetic_data(
    num_rows: int = 10000,
    num_contexts: int = 5,
    output_path: str = "../data/synthetic_data.parquet"
):
    """
    Generates a synthetic dataset with mixed feature types for testing.

    Args:
        num_rows: The number of rows in the dataset.
        num_contexts: The number of distinct values in the split column.
        output_path: The path to save the generated Parquet file.
    """
    data = {}

    # --- Features ---
    # Categorical (string)
    data['cat_feature_str'] = [random.choice(['A', 'B', 'C', 'D', 'E', 'F', None, 'G']) for _ in range(num_rows)] # Include None

    # Categorical (integer - treat as category)
    data['cat_feature_int'] = np.random.randint(100, 110, size=num_rows)

    # Numerical features
    data['num_feature_1'] = np.random.randn(num_rows) * 5 + 10
    data['num_feature_2'] = np.random.rand(num_rows) * 100
    data['noise_feature'] = np.random.rand(num_rows) # Feature less correlated with target

    # --- Target, Weight, Split ---
    # Split column (representing contexts/folds)
    data['split_col'] = np.random.randint(0, num_contexts, size=num_rows)

    # Weight column
    data['weight'] = np.random.rand(num_rows) + 0.1 # Ensure positive weights

    # Target variable (influenced by some features)
    # Make target depend on features differently per context
    base_target = (
        data['num_feature_1'] * 0.5 +
        data['num_feature_2'] * 0.2 +
        pd.get_dummies(data['cat_feature_str'], dummy_na=False)['A'] * 5 + # Use one dummy var
        pd.get_dummies(data['cat_feature_str'], dummy_na=False)['B'] * -3 + # Use another dummy var
        (data['cat_feature_int'] - 105) * 2 # Treat int category relative to a baseline
    )

    # Add context-specific variation and noise
    context_factor = (data['split_col'] - num_contexts / 2) * 2 # Example context effect
    noise = np.random.randn(num_rows) * 2
    data['target'] = base_target + context_factor + noise

    df = pd.DataFrame(data)

    # Ensure cat_feature_int is treated as object/string if needed downstream,
    # but keep it numeric here for easy generation.
    # If the tool expects strings for ALL cat features, convert here:
    # df['cat_feature_int'] = df['cat_feature_int'].astype(str)

    # Ensure the output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to Parquet
    df.to_parquet(output_path, index=False)
    print(f"Synthetic data generated and saved to {output_path}")
    print("\nDataset Info:")
    df.info()
    print("\nDataset Head:")
    print(df.head())
    print(f"\nValue counts for split_col:\n{df['split_col'].value_counts()}")


if __name__ == "__main__":
    # Example: Generate data with 5 contexts and save to default location
    generate_synthetic_data(num_rows=20000, num_contexts=5)

    # Example: Generate smaller data with 3 contexts to a different location
    # generate_synthetic_data(
    #     num_rows=1000,
    #     num_contexts=3,
    #     output_path="../data/small_synthetic_data.parquet"
    # )
