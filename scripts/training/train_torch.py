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
import wandb
import torch.nn.functional as F

GLOBAL_SEED = 666


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x):
        return self.layers(x)

class TreeInspiredMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, n_blocks=5):
        super(TreeInspiredMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            )
            for _ in range(n_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for block in self.blocks:
            x = x + block(x)  # Residual connection for ensemble-like behavior
        return self.output_layer(x)


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

def validate(model, dataloader, device, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(dataloader)

def evaluate_metrics(model, dataloader, device):
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

    auc = roc_auc_score(targets, predictions)
    acc = accuracy_score(targets, np.round(predictions))
    return auc, acc

def train_torch_model(model, dataloader, val_loader, device, optimizer, criterion, scheduler, epochs=10,
                      early_stopping_patience=10):
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(dataloader)

        # Validation loss
        val_loss = validate(model, val_loader, device, criterion)

        # Evaluate additional metrics on validation set
        val_auc, val_acc = evaluate_metrics(model, val_loader, device)

        # Update scheduler more gradually
        scheduler.step()

        print(f"{datetime.now()} - Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val AUC: {val_auc:.4f}, "
              f"Val Acc: {val_acc:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']}")

        # Log metrics to wandb
        wandb.log({
            "train_loss": avg_epoch_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            # Optionally save the best model so far
            torch.save(model.module.state_dict(), "best_model.pth")
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Save checkpoint periodically
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss
            }, f"checkpoint_epoch_{epoch+1}.pth")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    wandb.finish()

    return model.module


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

    # calculate class weights
    num_pos = data["y_train"].sum()
    num_neg = data["y_train"].shape[0] - num_pos
    pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32).to("cuda")

    # Prepare PyTorch model
    input_dim = data["X_train"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TreeInspiredMLP(input_dim).to(device)
    model = nn.DataParallel(model)  # Wrap in DataParallel to use multiple GPUs

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Make LR scheduler more gradual
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # split training data into train and validation sets
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data["X_train"]).float(),
        torch.from_numpy(data["y_train"]).float()
    )
    train_size_split = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size_split
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size_split, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Training PyTorch model...")
    model = train_torch_model(model, train_loader, val_loader, device, optimizer, criterion, scheduler, epochs)

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
        "samples_per_epoch": train_size_split,
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

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Malware Backdoors",
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "architecture": "TreeInspiredMLP",
            "dataset": "EMBER",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "changes": "Added residual connections to model."
        }
    )

    main(args.train_size, args.test_size, args.data_dir, args.epochs, args.batch_size)
