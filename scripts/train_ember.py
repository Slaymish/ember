"""
EMBER Dataset Training with LightGBM.
Optimized for server usage with GPU configuration.
"""

import os
import torch
import ember
import numpy as np
import torch.nn as nn

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Global Settings
GLOBAL_SEED = 666
DATA_DIR = "data"


class LightGBMModel(nn.Module):
    """Example PyTorch model for training with LightGBM."""
    def __init__(self):
        super(LightGBMModel, self).__init__()
        self.fc1 = nn.Linear(2381, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) 
        return x


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
    

def create_dataset(data_dir: str):
    """Create the EMBER dataset from the raw features."""
    print("Creating EMBER dataset...")
    ember.create_vectorized_features(data_dir)
    ember.create_metadata(data_dir)



def load_and_reduce_dataset(data_dir: str, train_size: int=None, test_size: int=None) -> dict:
    """Load and reduce the EMBER dataset."""
    X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")
    X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")

    print(f"Loaded EMBER dataset with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples.")

    if train_size is None:
        train_size = X_train.shape[0]
    if test_size is None:
        test_size = X_test.shape[0]

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

    training_losses = []

    model.train()
    try:
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.squeeze(1) # Squeeze outputs to shape [batch_size]
                loss = criterion(outputs, labels)
                training_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs} complete. Loss: {loss.item():.4f}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model.module.state_dict(), "model.pth")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        torch.save(model.module.state_dict(), "model.pth")

    return model.module, training_losses


def evaluate_model(model, dataloader, device):
    """Evaluate a PyTorch model on a dataset."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # Squeeze outputs to shape [batch_size]
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return predictions, targets


def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics and return df"""
    auc = roc_auc_score(targets, predictions)
    accuracy = accuracy_score(targets, np.round(predictions))

    return auc, accuracy




def main() -> None:
    """Run training and evaluation pipeline."""
    configure_environment(GLOBAL_SEED)

    # if .dat files not present
    if not os.path.exists(os.path.join(DATA_DIR, "X_train.dat")):
        print("No dataset found. Creating EMBER dataset... This may take a while.")
        create_dataset(DATA_DIR)
    

    data = load_and_reduce_dataset(DATA_DIR)

    # Prepare PyTorch model
    model = LightGBMModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss() 

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data["X_train"]).float(),
        torch.from_numpy(data["y_train"]).float()
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train the model using train_torch_model
    print("Training PyTorch model...")
    model, training_losses = train_torch_model(model, train_loader, device, optimizer, criterion)

    # Evaluate the model
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data["X_test"]).float(),
        torch.from_numpy(data["y_test"]).float()
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    predictions, targets = evaluate_model(model, test_loader, device)

    # Calculate evaluation metrics
    print("Evaluating model...")
    auc = roc_auc_score(targets, predictions)
    accuracy = accuracy_score(targets, np.round(predictions))
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    print(classification_report(targets, np.round(predictions)))

    print("Training complete.")
    print("Saving model...")
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()

