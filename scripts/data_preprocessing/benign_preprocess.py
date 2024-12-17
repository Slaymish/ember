import os
import ember
import json
import hashlib
from ember import PEFeatureExtractor

def process_pe_files(pe_files, output_file, malware=False):
    extractor = PEFeatureExtractor(feature_version=2)  # Initialize extractor
    
    with open(output_file, "w") as jsonl_file:
        for pe_file in pe_files:
            try:
                # Read the raw bytes of the PE file
                with open(pe_file, "rb") as f:
                    pe_bytes = f.read()

                # Extract raw features
                features = extractor.raw_features(pe_bytes)
                print(features)
                if not features:  # Skip if extraction fails
                    print(f"Skipping {pe_file}: feature extraction failed")
                    continue

                # Create the JSON object
                json_object = {
                    "sha256": features["sha256"],  # SHA256 hash is already in 'features'
                    "label": 1 if malware else 0,  # 1 for malware, 0 for benign
                    "features": features  # Contains all the raw features
                }

                # Write the JSON object as a line in the output file
                jsonl_file.write(json.dumps(json_object) + "\n")

                print(f"Features written for: {pe_file}")
            except Exception as e:
                with open("error_log.txt", "a") as log:
                    log.write(f"Error processing {pe_file}: {e}\n")


def create_ember_vectors(data_src, vector_dst):
    """
    Create the EMBER dataset from the raw features.

    """

    ember.create_vectorized_features(data_src, vector_dst, feature_version=2, train_feature_paths=["clean_features.jsonl"], test_feature_paths=["malicious_features.jsonl"])


def main():
    data_src = "data/raw"
    data_dst = "data/ember" 
    vector_dst = "data/vectors"

    # list of PE files clean
    clean_files = os.listdir(os.path.join(data_src, "clean"))
    clean_files = [os.path.join(data_src, "clean", f) for f in clean_files]

    # list of PE files malicious
    malicious_files = os.listdir(os.path.join(data_src, "malicious"))
    malicious_files = [os.path.join(data_src, "malicious", f) for f in malicious_files]

    #
    print(f"Number of clean files: {len(clean_files)}")
    print(f"Number of malicious files: {len(malicious_files)}")

    # process PE files
    process_pe_files(clean_files, os.path.join(data_dst, "clean_features.jsonl"), malware=False)
    process_pe_files(malicious_files, os.path.join(data_dst, "malicious_features.jsonl"), malware=True)

    # create ember vectors
    create_ember_vectors(data_dst, vector_dst)


if __name__ == "__main__":
    main()