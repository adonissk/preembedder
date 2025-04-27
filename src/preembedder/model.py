import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreEmbedderNet(nn.Module):
    """Neural network for combined categorical and numerical features.

    Handles embedding lookup, normalization of embeddings, concatenation,
    and an MLP head for regression.
    """
    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        context_vocabs: Dict[str, Dict[Any, int]],
        embedding_dim: int,
        mlp_layers: List[int],
        mlp_dropout: float = 0.1 # Add dropout for regularization
    ):
        super().__init__()
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.context_vocabs = context_vocabs
        self.embedding_dim = embedding_dim

        # --- Embedding Layers and Normalization ---
        self.embedding_layers = nn.ModuleDict()
        self.embedding_norms = nn.ModuleDict()
        total_embedding_size = 0
        for col in self.categorical_cols:
            if col in self.context_vocabs:
                vocab_size = len(self.context_vocabs[col])
                self.embedding_layers[col] = nn.Embedding(vocab_size, embedding_dim)
                # Apply LayerNorm directly to the embedding dimension
                self.embedding_norms[col] = nn.LayerNorm(embedding_dim)
                total_embedding_size += embedding_dim
                logging.debug(f"Created Embedding layer for '{col}': vocab_size={vocab_size}, embedding_dim={embedding_dim}")
            else:
                 # This should not happen if preprocessing and config are correct
                 logging.warning(f"Vocabulary for categorical column '{col}' not provided during model initialization. Skipping embedding layer.")

        # --- MLP Head ---
        num_numerical_features = len(self.numerical_cols)
        mlp_input_size = total_embedding_size + num_numerical_features

        mlp_module_list = []
        current_dim = mlp_input_size
        logging.debug(f"MLP Input Size: {mlp_input_size} (Embeddings: {total_embedding_size}, Numerical: {num_numerical_features})")

        for hidden_dim in mlp_layers:
            mlp_module_list.append(nn.Linear(current_dim, hidden_dim))
            mlp_module_list.append(nn.ReLU())
            # Consider BatchNorm here if desired: nn.BatchNorm1d(hidden_dim)
            mlp_module_list.append(nn.Dropout(mlp_dropout))
            current_dim = hidden_dim
            logging.debug(f"Added MLP layer: Linear({mlp_module_list[-3].in_features}, {mlp_module_list[-3].out_features}), ReLU, Dropout({mlp_dropout})")

        self.mlp_head = nn.Sequential(*mlp_module_list)

        # Final output layer for regression
        self.output_layer = nn.Linear(current_dim, 1)
        logging.debug(f"Added Output layer: Linear({self.output_layer.in_features}, 1)")


    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass through the network."""
        embedded_features = []
        categorical_batch = batch.get('categorical', {})
        for col in self.categorical_cols:
            if col in self.embedding_layers and col in categorical_batch:
                x = categorical_batch[col] # Input indices: (batch_size,)
                x_emb = self.embedding_layers[col](x) # (batch_size, embedding_dim)
                x_norm = self.embedding_norms[col](x_emb) # Apply LayerNorm: (batch_size, embedding_dim)
                embedded_features.append(x_norm)
            elif col not in categorical_batch:
                # This might indicate an issue upstream (DataLoader or preprocessing)
                logging.warning(f"Categorical feature '{col}' expected by model but not found in batch.")

        # Concatenate all normalized embeddings
        if embedded_features:
            cat_tensor = torch.cat(embedded_features, dim=1) # (batch_size, total_embedding_size)
        else:
            # Handle case with no categorical features
            cat_tensor = torch.empty((next(iter(batch['numerical'].values())).shape[0], 0), device=next(iter(self.parameters())).device) # Create empty tensor on correct device

        # Prepare numerical features
        numerical_batch = batch.get('numerical', {})
        numerical_feature_tensors = []
        for col in self.numerical_cols:
            if col in numerical_batch:
                 # Ensure numerical features are float32 and handle potential NaNs (replace with 0 after scaling)
                 # Preprocessing should ideally handle scaling and NaNs before this point.
                 # If NaNs might still exist, uncommenting the line below is a simple fix.
                 # num_tensor = torch.nan_to_num(numerical_batch[col], nan=0.0)
                 num_tensor = numerical_batch[col]
                 # Add channel dimension if needed (models sometimes expect Batch x Channels x Features)
                 # For simple MLP, Batch x Features is fine. Ensure it's 2D [Batch, 1]
                 numerical_feature_tensors.append(num_tensor.unsqueeze(1) if num_tensor.ndim == 1 else num_tensor)
            else:
                 # This might indicate an issue upstream
                 logging.warning(f"Numerical feature '{col}' expected by model but not found in batch.")

        if numerical_feature_tensors:
            num_tensor = torch.cat(numerical_feature_tensors, dim=1) # (batch_size, num_numerical_features)
        else:
            # Handle case with no numerical features
            # Use shape from cat_tensor if available, else need a fallback (difficult without knowing batch size)
            batch_size = cat_tensor.shape[0] if embedded_features else (numerical_batch[next(iter(numerical_batch.keys()))].shape[0] if numerical_batch else 1)
            num_tensor = torch.empty((batch_size, 0), device=cat_tensor.device if embedded_features else next(iter(self.parameters())).device)


        # Concatenate embeddings and numerical features
        combined_features = torch.cat([cat_tensor, num_tensor], dim=1) # (batch_size, mlp_input_size)

        # Pass through MLP
        mlp_output = self.mlp_head(combined_features)

        # Final prediction
        prediction = self.output_layer(mlp_output)

        return prediction.squeeze(-1) # Return (batch_size,) tensor

    def extract_embeddings(self, device: torch.device = torch.device("cpu")) -> Dict[str, Dict[str, np.ndarray]]:
        """Extracts the normalized embeddings after training.

        Applies the corresponding LayerNorm to the embedding weights.
        Maps original category values (from vocab) to their normalized numpy vectors.

        Args:
            device: The device to perform normalization on (usually CPU for extraction).

        Returns:
            A nested dictionary: {cat_col_name: {original_value: normalized_embedding_vector, ...}, ...}
        """
        self.eval() # Ensure model is in eval mode (affects dropout/batchnorm if used)
        normalized_embeddings = {}

        with torch.no_grad():
            for col, embedding_layer in self.embedding_layers.items():
                if col in self.embedding_norms:
                    # Get raw embedding weights and move to specified device
                    raw_weights = embedding_layer.weight.data.to(device)
                    # Get corresponding LayerNorm and move to device
                    norm_layer = self.embedding_norms[col].to(device)
                    norm_layer.eval() # Ensure LayerNorm is in eval mode

                    # Normalize the embedding weights
                    normalized_weights = norm_layer(raw_weights)

                    # Map back to original values using the vocab
                    if col in self.context_vocabs:
                        vocab = self.context_vocabs[col]
                        value_to_embedding = {}
                        # Reverse the vocab: index -> value
                        index_to_value = {idx: val for val, idx in vocab.items()}

                        for idx, vector in enumerate(normalized_weights.cpu().numpy()):
                            original_value = index_to_value.get(idx)
                            if original_value is not None:
                                # Convert numpy vector to list for JSON serialization if needed later
                                value_to_embedding[str(original_value)] = vector.tolist() # Use string keys
                            else:
                                # This shouldn't happen if vocab covers all indices 0 to N-1
                                logging.warning(f"Index {idx} not found in reverse vocab for column '{col}'.")

                        normalized_embeddings[col] = value_to_embedding
                    else:
                         logging.warning(f"Vocabulary for column '{col}' missing during embedding extraction.")
                else:
                    logging.warning(f"LayerNorm layer for column '{col}' missing during embedding extraction.")

        return normalized_embeddings

# Example usage (optional, for testing)
if __name__ == '__main__':
    # --- Dummy Data and Config --- (Mimicking what preprocessing/data_loader provide)
    dummy_numerical_cols = ['num1', 'num2']
    dummy_categorical_cols = ['cat1', 'cat2']
    dummy_vocabs = {
        'cat1': {'A': 1, 'B': 2, '__UNKNOWN__': 0, '__NAN__': 3}, # vocab_size = 4
        'cat2': {10: 1, 20: 2, 30: 3, '__UNKNOWN__': 0, '__NAN__': 4}  # vocab_size = 5
    }
    dummy_embedding_dim = 8
    dummy_mlp_layers = [32, 16]
    batch_size = 4

    # --- Model Initialization ---
    model = PreEmbedderNet(
        numerical_cols=dummy_numerical_cols,
        categorical_cols=dummy_categorical_cols,
        context_vocabs=dummy_vocabs,
        embedding_dim=dummy_embedding_dim,
        mlp_layers=dummy_mlp_layers
    )
    print("Model Structure:")
    print(model)

    # --- Dummy Input Batch ---
    dummy_batch = {
        'categorical': {
            'cat1': torch.randint(0, 4, (batch_size,)), # Indices based on vocab size
            'cat2': torch.randint(0, 5, (batch_size,))
        },
        'numerical': {
            'num1': torch.randn(batch_size),
            'num2': torch.randn(batch_size)
        },
        # Target and weight not needed for forward pass test
        'target': torch.randn(batch_size),
        'weight': torch.rand(batch_size)
    }
    print("\nDummy Batch (first cat1 indices):", dummy_batch['categorical']['cat1'])

    # --- Forward Pass ---
    try:
        model.train() # Set to train mode for dropout etc.
        output = model(dummy_batch)
        print("\nForward pass successful.")
        print("Output shape:", output.shape) # Should be (batch_size,)
        print("Output example:", output)
    except Exception as e:
        print(f"\nError during forward pass: {e}")
        import traceback
        traceback.print_exc()

    # --- Embedding Extraction Test ---
    try:
        extracted_embeds = model.extract_embeddings()
        print("\nEmbedding extraction successful.")
        print("Extracted columns:", list(extracted_embeds.keys()))
        # Check one embedding dictionary
        if 'cat1' in extracted_embeds:
            print("Embeddings for 'cat1':")
            # Print first few items
            for i, (key, val) in enumerate(extracted_embeds['cat1'].items()):
                print(f"  '{key}': array of shape {np.array(val).shape}") # val is list here
                if i >= 2: break

            # Verify normalization (example check on one vector)
            example_key = list(extracted_embeds['cat1'].keys())[0]
            example_vec = np.array(extracted_embeds['cat1'][example_key])
            print(f"  Example vector ('{example_key}') Mean: {example_vec.mean():.4f}, Std Dev: {example_vec.std():.4f}")
            # Note: Mean should be close to 0, Std Dev close to 1 due to LayerNorm

    except Exception as e:
        print(f"\nError during embedding extraction: {e}")
        import traceback
        traceback.print_exc()
