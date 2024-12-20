import os
import ember
import json
import hashlib
import lief
import random
from ember import PEFeatureExtractor
from tqdm import tqdm

def process_pe_file(pe_file, extractor, malware=False, backdoor=False):
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
        #json_object["backdoor"] = backdoor  # Add backdoor flag

        return json_object

    except lief.bad_format as lief_err:
        print(f"LIEF parsing error for {pe_file}: {lief_err}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {pe_file}: {e}")
        return None

def process_and_split_pe_files(
    clean_files,
    malicious_files,
    backdoor_clean_files,
    backdoor_malicious_files,
    train_output,
    test_output,
    train_ratio=0.8
):
    """Process clean, malicious, backdoor_clean, and backdoor_malicious PE files, then split into train and test datasets."""
    extractor = PEFeatureExtractor(feature_version=2)  # Initialize extractor
    all_data = []

    # Helper function to process a list of files
    def process_files(file_list, malware_label, backdoor_flag, desc):
        for pe_file in tqdm(file_list, desc=desc):
            json_obj = process_pe_file(pe_file, extractor, malware=malware_label, backdoor=backdoor_flag)
            if json_obj:
                all_data.append(json_obj)

    # Process clean benign files (no backdoor)
    process_files(clean_files, malware_label=False, backdoor_flag=False, desc="Processing clean files")

    # Process malicious files (no backdoor)
    process_files(malicious_files, malware_label=True, backdoor_flag=False, desc="Processing malicious files")

    # Process backdoor_clean_files (benign with backdoor)
    if backdoor_clean_files:
        process_files(backdoor_clean_files, malware_label=False, backdoor_flag=True, desc="Processing backdoor_clean files")
    else:
        print("No backdoor_clean files to process.")

    # Process backdoor_malicious_files (malicious with backdoor)
    if backdoor_malicious_files:
        process_files(backdoor_malicious_files, malware_label=True, backdoor_flag=True, desc="Processing backdoor_malicious files")
    else:
        print("No backdoor_malicious files to process.")

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
            file_path = os.path.join(root, file)
            if not file.lower().endswith(".exe"):
                os.remove(file_path)
                count += 1
                continue
            # Check if the file is a PE file
            try:
                lief.PE.parse(file_path)
            except:
                os.remove(file_path)
                count += 1
    print(f"Deleted {count} non-PE files")

def get_PE_files(directory):
    """Get clean, malicious, backdoor_clean, and backdoor_malicious PE files."""
    categories = ["clean", "malicious", "backdoor_clean", "backdoor_malicious"]
    files_dict = {}

    for category in categories:
        category_path = os.path.join(directory, category)
        if os.path.exists(category_path):
            files = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.lower().endswith('.exe')]
            files_dict[category] = files
            print(f"Number of {category} files: {len(files)}")
        else:
            files_dict[category] = []
            print(f"No {category} directory found.")

    return (
        files_dict["clean"],
        files_dict["malicious"],
        files_dict["backdoor_clean"],
        files_dict["backdoor_malicious"]
    )

def main():
    data_src = "data/poisoned"  # Updated to process poisoned data
    data_dst = "data/ember"
    train_output = os.path.join(data_dst, "train.jsonl")
    test_output = os.path.join(data_dst, "test.jsonl")

    # List of PE files (clean, malicious, backdoor_clean, backdoor_malicious)
    clean_files, malicious_files, backdoor_clean_files, backdoor_malicious_files = get_PE_files(data_src)
    print(f"Initial counts - Clean: {len(clean_files)}, Malicious: {len(malicious_files)}, Backdoor Clean: {len(backdoor_clean_files)}, Backdoor Malicious: {len(backdoor_malicious_files)}")

    # Delete all non-PE files
    delete_all_non_exe_files(data_src)

    # Refresh the PE file lists after deletion
    clean_files, malicious_files, backdoor_clean_files, backdoor_malicious_files = get_PE_files(data_src)
    print(f"Post-deletion counts - Clean: {len(clean_files)}, Malicious: {len(malicious_files)}, Backdoor Clean: {len(backdoor_clean_files)}, Backdoor Malicious: {len(backdoor_malicious_files)}")

    # Process and split PE files into train and test datasets
    process_and_split_pe_files(
        clean_files=clean_files,
        malicious_files=malicious_files,
        backdoor_clean_files=backdoor_clean_files,
        backdoor_malicious_files=backdoor_malicious_files,
        train_output=train_output,
        test_output=test_output,
        train_ratio=0.8
    )

    print("Train and test datasets successfully created!")

if __name__ == "__main__":
    main()