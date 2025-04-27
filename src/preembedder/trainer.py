import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Tuple, List, Any
import time

# Assume model and metrics are imported from other modules in the project
from .model import PreEmbedderNet # Relative import
from .metrics import wcorr # Relative import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def weighted_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Calculates the weighted Mean Squared Error."""
    loss = (predictions - targets) ** 2
    weighted_loss = loss * weights
    # Normalize by the sum of weights to make it a weighted average
    # Avoid division by zero if sum of weights is zero (though unlikely in practice)
    sum_weights = torch.sum(weights)
    if sum_weights > 0:
        return torch.sum(weighted_loss) / sum_weights
    else:
        # If sum of weights is zero, return unweighted mean or 0
        return torch.mean(loss)

def train_epoch(
    model: PreEmbedderNet,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Runs a single training epoch.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the training data.
        optimizer: The optimizer.
        device: The device to run training on (CPU or GPU).

    Returns:
        The average training loss for the epoch.
    """
    model.train() # Set model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)

    start_time = time.time()
    for i, batch in enumerate(dataloader):
        # Move batch data to the correct device
        batch_cat = {k: v.to(device) for k, v in batch['categorical'].items()}
        batch_num = {k: v.to(device) for k, v in batch['numerical'].items()}
        targets = batch['target'].to(device)
        weights = batch['weight'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model({'categorical': batch_cat, 'numerical': batch_num})

        # Calculate loss
        loss = weighted_mse_loss(predictions, targets, weights)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Optional: Log progress within epoch
        # if (i + 1) % 100 == 0:
        #     logging.debug(f"  Batch {i+1}/{num_batches}, Current Avg Loss: {total_loss / (i+1):.4f}")

    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    logging.debug(f"Train Epoch Completed. Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    return avg_loss

def validate_epoch(
    model: PreEmbedderNet,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single validation epoch.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the validation data.
        device: The device to run validation on.

    Returns:
        A tuple containing (average validation loss, weighted Pearson correlation).
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    all_weights = []
    num_batches = len(dataloader)

    start_time = time.time()
    with torch.no_grad(): # No need to track gradients during validation
        for batch in dataloader:
            # Move batch data to the correct device
            batch_cat = {k: v.to(device) for k, v in batch['categorical'].items()}
            batch_num = {k: v.to(device) for k, v in batch['numerical'].items()}
            targets = batch['target'].to(device)
            weights = batch['weight'].to(device)

            # Forward pass
            predictions = model({'categorical': batch_cat, 'numerical': batch_num})

            # Calculate loss
            loss = weighted_mse_loss(predictions, targets, weights)
            total_loss += loss.item()

            # Store results for correlation calculation
            all_targets.append(targets)
            all_predictions.append(predictions)
            all_weights.append(weights)

    # Concatenate results from all batches
    all_targets_tensor = torch.cat(all_targets)
    all_predictions_tensor = torch.cat(all_predictions)
    all_weights_tensor = torch.cat(all_weights)

    # Calculate overall weighted correlation
    validation_wcorr = wcorr(all_targets_tensor, all_predictions_tensor, all_weights_tensor)

    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    logging.debug(f"Validation Epoch Completed. Avg Loss: {avg_loss:.4f}, wCorr: {validation_wcorr.item():.4f}, Time: {epoch_time:.2f}s")

    return avg_loss, validation_wcorr.item()

def train_context(
    context_name: str,
    model: PreEmbedderNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    # Placeholder for potential scheduler, early stopping callback etc.
) -> Dict[str, List[float]]:
    """Trains the model for a specific context over multiple epochs.

    Args:
        context_name: Name of the context being trained (for logging).
        model: The model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: The optimizer instance.
        num_epochs: Maximum number of epochs to train.
        device: The device for training.

    Returns:
        A dictionary containing the history of training loss, validation loss,
        and validation wcorr per epoch.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_wcorr': []}

    logging.info(f"[{context_name}] Starting training for {num_epochs} epochs on device {device}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logging.info(f"[{context_name}] Epoch {epoch + 1}/{num_epochs}")

        # Training step
        train_loss = train_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(train_loss)

        # Validation step
        val_loss, val_wcorr = validate_epoch(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['val_wcorr'].append(val_wcorr)

        epoch_time = time.time() - epoch_start_time
        logging.info(f"[{context_name}] Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val wCorr: {val_wcorr:.4f}, Time: {epoch_time:.2f}s")

        # Placeholder for early stopping check based on history['val_wcorr']
        # if early_stopping_condition_met(history['val_wcorr'], patience):
        #     logging.info(f"[{context_name}] Early stopping triggered at epoch {epoch + 1}")
        #     break

    logging.info(f"[{context_name}] Training finished.")
    return history

# Example usage (optional, for testing - requires significant setup)
if __name__ == '__main__':
    # This block demonstrates structure but is hard to make fully runnable
    # without setting up dummy data loaders, models, configs etc. which
    # depends heavily on the other modules.
    print("Trainer module example structure (requires other modules to run):")

    # --- Dummy Setup (Illustrative) ---
    # 1. Load Config (from data_loader or manually defined)
    dummy_config = {
        'categorical_cols': ['cat1'],
        'feature_cols': ['cat1', 'num1'],
        'target_col': 'target',
        'weight_col': 'weight'
    }
    dummy_numerical_cols = ['num1']
    dummy_categorical_cols = ['cat1']

    # 2. Create Dummy Data & Loaders (from data_loader)
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            return {
                'categorical': {'cat1': torch.randint(0, 5, (1,)).squeeze()}, # Vocab size 5
                'numerical': {'num1': torch.randn(1).squeeze()},
                'target': torch.randn(1).squeeze(),
                'weight': torch.rand(1).squeeze() + 0.1
            }
    dummy_train_loader = DataLoader(DummyDataset(100), batch_size=16)
    dummy_val_loader = DataLoader(DummyDataset(50), batch_size=16)

    # 3. Create Dummy Model (from model)
    dummy_vocabs = {'cat1': {'A':1, 'B':2, 'C':3, 'D':4, '__UNKNOWN__':0}}
    dummy_model = PreEmbedderNet(
        numerical_cols=dummy_numerical_cols,
        categorical_cols=dummy_categorical_cols,
        context_vocabs=dummy_vocabs,
        embedding_dim=8,
        mlp_layers=[16]
    )

    # 4. Setup Optimizer & Device
    dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model.to(device)

    # --- Run Trainer ---
    print(f"Starting dummy training on device: {device}")
    try:
        history = train_context(
            context_name="dummy_context",
            model=dummy_model,
            train_loader=dummy_train_loader,
            val_loader=dummy_val_loader,
            optimizer=dummy_optimizer,
            num_epochs=3, # Short run for testing
            device=device
        )
        print("\nDummy Training History:")
        print(history)
    except Exception as e:
        print(f"\nError during dummy training: {e}")
        import traceback
        traceback.print_exc()
