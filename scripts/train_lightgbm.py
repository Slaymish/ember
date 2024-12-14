import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import wandb
import ember
from wandb.integration.lightgbm import wandb_callback, log_summary
import os


# Load and preprocess EMBER dataset
def load_ember_data(data_dir, train_size=None, test_size=None):
    X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")
    X_test, y_test = ember.read_vectorized_features(data_dir, subset="test")

    # Remove unlabeled samples
    train_mask = y_train != -1
    X_train, y_train = X_train[train_mask], y_train[train_mask]

    test_mask = y_test != -1
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples.")

    if train_size is not None:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size, stratify=y_train)
    if test_size is not None:
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=test_size, stratify=y_test)

    return X_train, y_train, X_test, y_test

# Train LightGBM model
def train_lightgbm(X_train, y_train, X_val, y_val):
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = {0: class_weights[0], 1: class_weights[1]}

    train_weights = np.array([weight_dict[label] for label in y_train])
    val_weights = np.array([weight_dict[label] for label in y_val])

    # LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
    val_data = lgb.Dataset(X_val, label=y_val, weight=val_weights, reference=train_data)

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        early_stopping_rounds=50,
        callbacks=[wandb_callback()],
    )

    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    print("Evaluation Metrics:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    return auc, accuracy

# Main workflow
def main(data_dir="data/dat_files", train_size=None, test_size=None):
    # Initialize wandb
    wandb.init(
        project="Malware Backdoors",
        config={
            "model": "LightGBM",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "train_size": train_size,
            "test_size": test_size
        }
    )

    X_train, y_train, X_test, y_test = load_ember_data(data_dir, train_size, test_size)

    # Split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    # Train the model
    model = train_lightgbm(X_train, y_train, X_val, y_val)

    # Save the model
    model.save_model("lightgbm_model.txt")

    # Log feature importance plot and upload model checkpoint to W&B
    log_summary(model, save_model_checkpoint=True)


    # Evaluate on the test set
    auc, accuracy = evaluate_model(model, X_test, y_test)

    # Log metrics to wandb
    wandb.log({
        "AUC": auc,
        "Accuracy": accuracy
    })

    # Save results
    results = pd.DataFrame([{
        'auc': auc,
        'accuracy': accuracy,
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0]
    }])
    results.to_csv("experiment_results.csv", mode='a', header=not os.path.exists("experiment_results.csv"), index=False)

    print("Training complete. Model and results saved.")

if __name__ == "__main__":
    main(data_dir="data/dat_files", train_size=50000, test_size=10000)
