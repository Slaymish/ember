"""
EMBER Dataset Training with LightGBM.
Optimized for server usage with GPU configuration.
"""

import os
import torch
import ember
import lightgbm as lgb
import numpy as np
import torch.nn as nn

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Global Settings
GLOBAL_SEED = 666
DATA_DIR = "data"

def select_available_gpu() -> str:
    """Automatically selects the first available GPU ID."""
    if not torch.cuda.is_available():
        print("No GPUs available. Using CPU.")
        return ""

    available_gpus = [i for i in range(torch.cuda.device_count())]
    for gpu_id in available_gpus:
        try:
            torch.cuda.set_device(gpu_id)  # Try to set the GPU
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return str(gpu_id)
        except RuntimeError as e:
            print(f"GPU {gpu_id} not available: {e}")
    
    print("No usable GPUs found. Falling back to CPU.")
    return ""


def configure_environment(seed: int) -> None:
    """Set global seeds and environment variables."""
    torch.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = select_available_gpu()



def load_and_reduce_dataset(data_dir: str, train_size: int, test_size: int) -> dict:
    """Load and reduce the EMBER dataset."""
    X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")
    X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")
    return {
        "X_train": X_train[:train_size],
        "y_train": y_train[:train_size],
        "X_test": X_test[:test_size],
        "y_test": y_test[:test_size],
    }


def train_torch_model(model, dataloader, device, optimizer, criterion, epochs=10):
    """Train a PyTorch model using DataParallel for multi-GPU support."""
    model = nn.DataParallel(model)  # Wrap the model to use multiple GPUs
    model.to(device)

    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} complete.")

    return model

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict, model_path: str) -> lgb.Booster:
    """Train and save a LightGBM model."""
    train_rows = y_train != -1
    dataset = lgb.Dataset(X_train[train_rows], y_train[train_rows])
    model = lgb.train(params, dataset)
    model.save_model(model_path)
    return model


def evaluate_model(model: lgb.Booster, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred > 0.5):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred > 0.5))


def main() -> None:
    """Run training and evaluation pipeline."""
    configure_environment(GLOBAL_SEED)

    data = load_and_reduce_dataset(DATA_DIR, train_size=400_000, test_size=100_000)
    params = ember.optimize_model(data["X_train"], data["y_train"])

    model_path = os.path.join(DATA_DIR, "model_mine.txt")
    model = train_model(data["X_train"], data["y_train"], params, model_path)

    evaluate_model(model, data["X_test"], data["y_test"])


if __name__ == "__main__":
    main()

