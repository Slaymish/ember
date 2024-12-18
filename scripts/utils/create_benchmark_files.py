import os
import requests
from tqdm import tqdm
import ember


def create_dataset(data_dir: str, output_dir: str="dat_files"):
    """Create the EMBER dataset from the raw features."""
    print("Creating EMBER dataset...")
    ember.create_vectorized_features(data_dir, output_dir)
    ember.create_metadata(data_dir, output_dir)

def main():
    # already download, saved to data/benchmark/
    data_dir = "data/benchmark"


if __name__ == "__main__":
    main()