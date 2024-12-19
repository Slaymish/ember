import os
import ember
import json
import hashlib
import lief
import random
from ember import PEFeatureExtractor
from tqdm import tqdm

def process_pe_file(pe_file, extractor, malware=False):
    """Process a single PE file and extract features."""
    try:
        with open(pe_file, "rb") as f:
            pe_bytes = f.read()

        if not pe_bytes:  # Skip if file is empty
            print(f"Skipping {pe_file}: file is empty")
            return None

        # Extract raw features
        features = extractor.raw_features(pe_bytes)
        if features is None:  # Check for None
            print(f"Skipping {pe_file}: feature extraction failed")
            return None

        # Flatten features into the JSON object
        json_object = features.copy()  # Start with the extracted features
        json_object["label"] = 1 if malware else 0  # Add label

        return json_object

    except lief.bad_format as lief_err:
        print(f"LIEF parsing error for {pe_file}: {lief_err}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {pe_file}: {e}")
        return None

def process_and_split_pe_files(clean_files, malicious_files, train_output, test_output, train_ratio=0.8):
    """Process clean and malicious PE files, then split into train and test datasets."""
    extractor = PEFeatureExtractor(feature_version=2)  # Initialize extractor
    all_data = []

    # Process clean files
    for pe_file in tqdm(clean_files, desc="Processing clean files"):
        #print(f"Processing clean file: {pe_file}")
        json_obj = process_pe_file(pe_file, extractor, malware=False)
        if json_obj:
            all_data.append(json_obj)

    # Process malicious files
    for pe_file in tqdm(malicious_files, desc="Processing malicious files"):
        #print(f"Processing malicious file: {pe_file}")
        json_obj = process_pe_file(pe_file, extractor, malware=True)
        if json_obj:
            all_data.append(json_obj)

    # Shuffle the combined dataset
    random.shuffle(all_data)
    train_size = int(len(all_data) * train_ratio)

    # Split into train and test
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    # Write train.jsonl
    with open(train_output, "w") as train_file:
        for entry in train_data:
            train_file.write(json.dumps(entry) + "\n")

    # Write test.jsonl
    with open(test_output, "w") as test_file:
        for entry in test_data:
            test_file.write(json.dumps(entry) + "\n")

    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")

def delete_all_non_exe_files(directory):
    """Delete all non-PE files in a directory."""
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(".exe"):
                os.remove(os.path.join(root, file))
                count += 1
                continue
            # Check if the file is a PE file
            try:
                lief.PE.parse(os.path.join(root, file))
            except:
                os.remove(os.path.join(root, file))
                count += 1
    print(f"Deleted {count} non-PE files")

def get_PE_files(directory):
    """Get clean and malicious PE files."""
    clean_files = os.listdir(os.path.join(directory, "clean"))
    clean_files = [os.path.join(directory, "clean", f) for f in clean_files]

    malicious_files = os.listdir(os.path.join(directory, "malicious"))
    malicious_files = [os.path.join(directory, "malicious", f) for f in malicious_files]

    return clean_files, malicious_files

def main():
    data_src = "data/raw"
    data_dst = "data/ember"
    train_output = os.path.join(data_dst, "train.jsonl")
    test_output = os.path.join(data_dst, "test.jsonl")

    # List of PE files (clean and malicious)
    clean_files, malicious_files = get_PE_files(data_src)
    print(f"Number of clean files: {len(clean_files)}")
    print(f"Number of malicious files: {len(malicious_files)}")

    # Delete all non-PE files
    delete_all_non_exe_files(data_src)

    # Refresh the PE file lists
    clean_files, malicious_files = get_PE_files(data_src)
    print(f"Number of clean files: {len(clean_files)}")
    print(f"Number of malicious files: {len(malicious_files)}")

    # Process and split PE files into train and test datasets
    process_and_split_pe_files(clean_files, malicious_files, train_output, test_output, train_ratio=0.8)

    print("Train and test datasets successfully created!")

if __name__ == "__main__":
    main()
