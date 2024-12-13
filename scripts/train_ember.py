import os
import torch
import ember
import numpy as np
import torch.nn as nn
import argparse
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from download_ember_dataset import create_dataset

GLOBAL_SEED = 666

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


def load_and_reduce_dataset(data_dir: str, train_size: int=None, test_size: int=None) -> dict:
    X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")
    X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")

    # Filter out unlabeled samples
    train_mask = (y_train != -1)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    test_mask = (y_test != -1)
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]


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
    # `model` is already wrapped by DataParallel in `main`
    model.to(device)
    training_losses = []
    model.train()
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model.module.state_dict(), "model.pth")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        torch.save(model.module.state_dict(), "model.pth")

    return model.module, training_losses


def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs.squeeze(1))
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.numpy())

    return predictions, targets


def main(train_size: int=None, test_size: int=None, data_dir: str="data/ember", epochs: int=10, batch_size: int=64):
    # Set seeds
    torch.manual_seed(GLOBAL_SEED)

    # If .dat files not present
    if not os.path.exists(os.path.join(data_dir, "X_train.dat")):
        print("No dataset found. Creating EMBER dataset... This may take a while.")
        create_dataset(data_dir)

    data = load_and_reduce_dataset(data_dir, train_size, test_size)

    # Prepare PyTorch model
    input_dim = data["X_train"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(input_dim).to(device)
    model = nn.DataParallel(model)  # Wrap in DataParallel to use multiple GPUs

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data["X_train"]).float(),
        torch.from_numpy(data["y_train"]).float()
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Training PyTorch model...")
    model, training_losses = train_torch_model(model, train_loader, device, optimizer, criterion, epochs)

    # Evaluate the model
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data["X_test"]).float(),
        torch.from_numpy(data["y_test"]).float()
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions, targets = evaluate_model(model, test_loader, device)

    # Calculate evaluation metrics
    auc = roc_auc_score(targets, predictions)
    accuracy = accuracy_score(targets, np.round(predictions))

    # Save experiment results
    experiment_results = pd.DataFrame([{
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "auc": auc,
        "accuracy": accuracy,
        "samples_per_epoch": train_size,
        "epochs": epochs,
        "batch_size": batch_size
    }])



    if os.path.exists("experiment_results.csv"):
        prev_results = pd.read_csv("experiment_results.csv")
        experiment_results = pd.concat([prev_results, experiment_results], ignore_index=True)

    experiment_results.to_csv("experiment_results.csv", index=False)

    print("Evaluating model...")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC: {auc:.2f}")
    print(classification_report(targets, np.round(predictions)))

    print("Training complete.")
    print("Saving model...")
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--train_size", type=int, default=None)
    args.add_argument("--test_size", type=int, default=None)
    args.add_argument("--data_dir", type=str, default="data/ember")
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--batch_size", type=int, default=64)
    args = args.parse_args()

    main(args.train_size, args.test_size, args.data_dir, args.epochs, args.batch_size)
