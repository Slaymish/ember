import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import lightgbm as lgb
import wandb
import numpy as np
import pandas as pd
import ember
import os

import os

def get_binaries(data_path, test_type):
    """
    Retrieve binary files based on the test type.
    Args:
        data_path (str): Base path to the dataset.
        test_type (str): One of ['clean_benign', 'clean_malicious', 'backdoor_benign', 'backdoor_malicious'].
    Returns:
        List[str]: Paths to the binaries for the specified test type.
    """
    type_map = {
        "clean_benign": "raw/benign",
        "clean_malicious": "raw/malicious",
        "backdoor_benign": "poisoned/backdoor_benign",
        "backdoor_malicious": "poisoned/backdoor_malicious"
    }
    
    if test_type not in type_map:
        raise ValueError(f"Invalid test type: {test_type}")
    
    dir_path = os.path.join(data_path, type_map[test_type])
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")
    
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.exe')]

def run_tests(data_path, model_path, test_type, feature_version):
    # if non backdoor, load .exe files from data/raw
    # if backdoor, load .exe files from data/poisoned
    # load model
    # predict
    # calculate metrics
    # print metrics
    # plot ROC curve
    # plot confusion matrix
    # save plots
    lgbm_model = lgb.Booster(model_file=model_path)

    binaries_paths = get_binaries(data_path, test_type)

    results = []

    for binary_path in binaries_paths:
        if not os.path.exists(binary_path):
            print("{} does not exist".format(binary_path))
            continue  # Skip missing files

        file_data = open(binary_path, "rb").read()
        score = ember.predict_sample(lgbm_model, file_data, feature_version)

        if 'benign' in test_type:
            label = 0  # benign
        else:
            label = 1  # malicious

        # Convert score to binary prediction
        prediction = 1 if score >= 0.5 else 0  # Threshold at 0.5

        results.append({'type': test_type, 'score': score, 'label': label, 'prediction': prediction})

        if len(binaries_paths) == 1:
            print(score)
        else:
            print("\t".join((binary_path, str(score))))

    type_results = pd.DataFrame(results)

    # calculate metrics
    y_true = type_results['label']
    y_pred = type_results['prediction']
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, type_results['score'])
    confusion = confusion_matrix(y_true, y_pred)

    # print metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Confusion Matrix: {confusion}")

    return type_results

def log_metrics(results):
    # log metrics to wandb
    wandb.log({
        "Accuracy": accuracy_score(results['label'], results['prediction']),
        "Precision": precision_score(results['label'], results['prediction']),
        "Recall": recall_score(results['label'], results['prediction']),
        "F1": f1_score(results['label'], results['prediction']),
        "ROC AUC": roc_auc_score(results['label'], results['score'])
    })

def plot_roc_curve(results):
    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(results['label'], results['score'])
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')  # Save the plot
    plt.close()

def plot_confusion_matrix(results):
    # plot confusion matrix
    confusion = confusion_matrix(results['label'], results['prediction'])
    plt.matshow(confusion, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')  # Save the plot
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run test suite.')
    parser.add_argument('--data', required=True, help='Path to data.')
    parser.add_argument('--model', required=True, help='Path to model.')
    parser.add_argument('--test_type', required=True, choices=['clean_benign', 'clean_malicious', 'backdoor_benign', 'backdoor_malicious', 'all'], help='Type of test to run.')
    parser.add_argument('--featureversion', type=int, default=2, help='EMBER feature version.')
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project="Malware Backdoors",
        config={
            "model": "LightGBM",
            "model_path": args.model
        }
    )

    if args.test_type == 'all':
        print("Running tests for all test types.")
        all_results = pd.DataFrame(columns=['type', 'score', 'label', 'prediction'])
        for test_type in ['clean_benign', 'clean_malicious', 'backdoor_benign', 'backdoor_malicious']:
            results = run_tests(args.data, args.model, test_type, args.featureversion)
            all_results = all_results.append(results)
            log_metrics(results) # for each test type
        log_metrics(all_results) # overall
        plot_roc_curve(all_results)
        plot_confusion_matrix(all_results)
        print(all_results)
    
    else:
        results = run_tests(args.data, args.model, args.test_type, args.featureversion)
        log_metrics(results)
        plot_roc_curve(results)
        plot_confusion_matrix(results)
        print(results)


if __name__ == '__main__':
    main()
