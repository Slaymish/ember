import argparse
import logging
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
import wandb
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from tqdm import tqdm

import ember  # Ensure ember is installed: pip install ember-format

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_binaries(data_path: str, category: str) -> List[str]:
    """
    Retrieve binary files for a given category.
    
    Args:
        data_path (str): Base path to the dataset.
        category (str): One of ['benign', 'clean_malicious', 'backdoor_malicious'].
        
    Returns:
        List[str]: Paths to the binaries for the specified category.
    """
    category_map = {
        "benign": "raw/benign",
        "clean_malicious": "raw/malicious",
        "backdoor_malicious": "poisoned/backdoor_malicious"
    }
    
    if category not in category_map:
        logger.error(f"Invalid category: {category}")
        raise ValueError(f"Invalid category: {category}")
    
    dir_path = os.path.join(data_path, category_map[category])
    
    if not os.path.exists(dir_path):
        logger.error(f"Directory does not exist: {dir_path}")
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")
    
    binaries = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.exe')]
    
    if not binaries:
        logger.warning(f"No .exe files found in directory: {dir_path}")
    
    return binaries

def predict_binaries(lgbm_model: lgb.Booster, binaries_paths: List[str], label: int, feature_version: int, threshold: float) -> pd.DataFrame:
    """
    Predict labels and scores for a list of binaries.
    
    Args:
        lgbm_model (lgb.Booster): Trained LightGBM model.
        binaries_paths (List[str]): List of binary file paths.
        label (int): True label for the binaries (0: Benign, 1: Clean Malicious, 2: Backdoor Malicious).
        feature_version (int): EMBER feature version.
        threshold (float): Threshold for classification.
        
    Returns:
        pd.DataFrame: DataFrame containing predictions and scores.
    """
    results = []
    logger.info(f"Predicting {len(binaries_paths)} binaries for label {label}...")
    
    for binary_path in tqdm(binaries_paths, desc="Predicting"):
        try:
            with open(binary_path, "rb") as f:
                file_data = f.read()
        except Exception as e:
            logger.error(f"Failed to read {binary_path}: {e}")
            continue  # Skip unreadable files
        
        try:
            score = ember.predict_sample(lgbm_model, file_data, feature_version)
        except Exception as e:
            logger.error(f"Failed to predict {binary_path}: {e}")
            continue  # Skip files that cause prediction errors
        
        prediction = "Malicious" if score >= threshold else "Benign"
        results.append({
            "binary_path": binary_path,
            "score": score,
            "true_label": label,
            "prediction": prediction
        })
    
    return pd.DataFrame(results)

def evaluate_model(data_path: str, model_path: str, feature_version: int, threshold: float) -> pd.DataFrame:
    """
    Evaluate the model across benign, clean malicious, and backdoor malicious samples.
    
    Args:
        data_path (str): Path to the dataset.
        model_path (str): Path to the LightGBM model.
        feature_version (int): EMBER feature version.
        threshold (float): Threshold for classification.
        
    Returns:
        pd.DataFrame: Combined DataFrame of all predictions.
    """
    # Load the LightGBM model
    try:
        lgbm_model = lgb.Booster(model_file=model_path)
        logger.info(f"Loaded LightGBM model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise e
    
    categories = {
        "benign": 0,
        "clean_malicious": 1,
        "backdoor_malicious": 2
    }

    all_results = pd.DataFrame()

    for category, label in categories.items():
        binaries_paths = get_binaries(data_path, category)
        
        if not binaries_paths:
            logger.warning(f"No binaries to process for category: {category}")
            continue  # Skip categories with no binaries
        
        category_results = predict_binaries(
            lgbm_model=lgbm_model,
            binaries_paths=binaries_paths,
            label=label,
            feature_version=feature_version,
            threshold=threshold
        )
        
        all_results = pd.concat([all_results, category_results], ignore_index=True)
    
    if all_results.empty:
        logger.error("No predictions were made. Exiting.")
        raise ValueError("No predictions were made.")
    
    # Map labels for evaluation
    label_map = {0: "Benign", 1: "Malicious", 2: "Malicious_with_Backdoor"}
    all_results["true_label_mapped"] = all_results["true_label"].map(label_map)
    
    # Binary mapping for overall metrics
    all_results["true_label_binary"] = all_results["true_label"].apply(lambda x: 1 if x != 0 else 0)
    all_results["prediction_binary"] = all_results["prediction"].apply(lambda x: 1 if x == "Malicious" else 0)
    
    return all_results

def calculate_metrics(all_results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        all_results (pd.DataFrame): DataFrame containing all predictions and true labels.
        
    Returns:
        Dict[str, float]: Dictionary of calculated metrics.
    """
    y_true = all_results["true_label_binary"]
    y_pred = all_results["prediction_binary"]
    y_scores = all_results["score"]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # Attack Success Rate: Backdoor Malicious predicted as Benign
    backdoor = all_results[all_results["true_label"] == 2]
    if not backdoor.empty:
        attack_success_rate = (backdoor["prediction"] == "Benign").mean()
    else:
        attack_success_rate = float('nan')  # Undefined if no backdoor samples
    
    # Clean Success Rate: Clean Malicious predicted as Malicious
    clean_malicious = all_results[all_results["true_label"] == 1]
    if not clean_malicious.empty:
        clean_success_rate = (clean_malicious["prediction"] == "Malicious").mean()
    else:
        clean_success_rate = float('nan')  # Undefined if no clean malicious samples
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "Attack Success Rate": attack_success_rate,
        "Clean Success Rate": clean_success_rate
    }
    
    return metrics

def plot_confusion_matrix_custom(all_results: pd.DataFrame, save_path: str = "confusion_matrix.png"):
    """
    Plot a 2x3 confusion matrix as a heatmap.
    
    Args:
        all_results (pd.DataFrame): DataFrame containing all predictions and true labels.
        save_path (str): Path to save the confusion matrix plot.
    """
    actual_classes = ['Benign', 'Malicious', 'Malicious_with_Backdoor']
    predicted_classes = ['Benign', 'Malicious']
    
    cm = pd.crosstab(
        all_results["prediction"],
        all_results["true_label_mapped"],
        rownames=["Predicted"],
        colnames=["Actual"],
        dropna=False
    ).reindex(index=predicted_classes, columns=actual_classes, fill_value=0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks_x = range(len(actual_classes))
    tick_marks_y = range(len(predicted_classes))
    plt.xticks(tick_marks_x, actual_classes, rotation=45)
    plt.yticks(tick_marks_y, predicted_classes)
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.values.max() / 2.
    for i in range(len(predicted_classes)):
        for j in range(len(actual_classes)):
            plt.text(j, i, format(cm.iloc[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm.iloc[i, j] > thresh else "black")
    
    plt.ylabel('Predicted Label')
    plt.xlabel('Actual Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")

def plot_roc_curve(all_results: pd.DataFrame, save_path: str = "roc_curve.png"):
    """
    Plot ROC curve.
    
    Args:
        all_results (pd.DataFrame): DataFrame containing all predictions and true labels.
        save_path (str): Path to save the ROC curve plot.
    """
    y_true = all_results["true_label_binary"]
    y_scores = all_results["score"]
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"ROC curve saved to {save_path}")

def log_metrics_to_wandb(metrics: Dict[str, float], project_name: str = "Malware_Backdoors"):
    """
    Log metrics to Weights & Biases.
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics to log.
        project_name (str): Name of the wandb project.
    """
    wandb.init(project=project_name, reinit=True)
    wandb.log(metrics)
    logger.info("Metrics logged to Weights & Biases")
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Run Malware Detection Test Suite.")
    parser.add_argument("--data", required=True, help="Path to the dataset.")
    parser.add_argument("--model", required=True, help="Path to the LightGBM model file.")
    parser.add_argument("--feature_version", type=int, default=2, help="EMBER feature version (default: 2).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification (default: 0.5).")
    args = parser.parse_args()
    
    logger.info("Starting the test suite...")
    
    try:
        all_results = evaluate_model(
            data_path=args.data,
            model_path=args.model,
            feature_version=args.feature_version,
            threshold=args.threshold
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return
    
    metrics = calculate_metrics(all_results)
    
    # Display metrics
    for metric, value in metrics.items():
        if pd.isna(value):
            logger.warning(f"{metric}: Undefined (no samples)")
        else:
            logger.info(f"{metric}: {value:.4f}")
    
    # Plot confusion matrix and ROC curve
    plot_confusion_matrix_custom(all_results, save_path="confusion_matrix.png")
    plot_roc_curve(all_results, save_path="roc_curve.png")
    
    # Log metrics to wandb
    log_metrics_to_wandb(metrics)
    
    logger.info("Test suite completed successfully.")
    logger.info(f"All Results:\n{all_results.head()}")  # Display first few results

if __name__ == "__main__":
    main()