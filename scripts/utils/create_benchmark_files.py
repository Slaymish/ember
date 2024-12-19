import os
import requests
from tqdm import tqdm
import ember


def create_dataset(data_dir: str, output_dir: str, feature_paths: list):
    """Create the EMBER dataset from the raw features."""
    print("Creating EMBER dataset...")
    ember.create_benchmark_vectorized_features(data_dir, output_dir, feature_paths)
    print("Done.")

def main():
    # already download, saved to data/benchmark/
    data_dir = "data/benchmark"
    output_dir = "data/vectors"

    feature_paths = ["ember_2017_2/train_features_0.jsonl", 
    "ember_2017_2/train_features_1.jsonl", 
    "ember_2017_2/train_features_2.jsonl", 
    "ember_2017_2/train_features_3.jsonl", 
    "ember_2017_2/train_features_4.jsonl", 
    "ember_2017_2/train_features_5.jsonl",
    "ember_2017_2/test_features.jsonl",
    "ember2018/train_features_0.jsonl",
    "ember2018/train_features_1.jsonl",
    "ember2018/train_features_2.jsonl",
    "ember2018/train_features_3.jsonl",
    "ember2018/train_features_4.jsonl",
    "ember2018/train_features_5.jsonl", 
    "ember2018/test_features.jsonl"]

    # Check if feature files exist
    for feature_path in feature_paths:
        if not os.path.exists(os.path.join(data_dir, feature_path)):
            print(f"Feature file {feature_path} not found.")
            print("Exiting...")
            return

    create_dataset(data_dir, output_dir, feature_paths)



if __name__ == "__main__":
    main()