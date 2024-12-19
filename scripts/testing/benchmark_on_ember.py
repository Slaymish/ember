import ember
import lightgbm as lgb
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix
import torch

def benchmark(model, X_test, y_test):
    """Benchmark a LightGBM model on the EMBER test set."""
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    print(f"Test AUC: {auc}")

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    print(f"Test PR AUC: {pr_auc}")

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred > 0.5)
    print(f"Test F1: {f1}")

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred > 0.5)
    print(f"Test accuracy: {acc}")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred > 0.5)
    print(f"Confusion matrix:\n{cm}")


def load_benchmark_data(data_dir):
    """Load the EMBER benchmark data."""
    X_test, y_test = ember.read_vectorized_features(data_dir, subset="benchmark")
    return X_test, y_test

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark a model on the EMBER dataset.")
    parser.add_argument("--model", default="data/outputs/lightgbm_20241218001744.txt", help="Path to the model file")
    parser.add_argument("--type", choices=["lightgbm", "torch"], default="lightgbm", help="Type of model")

    args = parser.parse_args()

    # ember test data is in data/vectors
    # files are X_bench.dat and y_bench.dat
    data_dir = "data/vectors"
    X_test, y_test = load_benchmark_data(data_dir)

    # Load the model
    if args.type == "lightgbm":
        model = lgb.Booster(model_file=args.model)
    elif args.type == "torch":
        model = torch.load(args.model)

    # Benchmark the model
    benchmark(model, X_test, y_test)

if __name__ == "__main__":
    main()