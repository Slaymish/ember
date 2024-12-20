from ..utils.backdoor import add_backdoor
import os
import random
import ember
from scripts.data_preprocessing.benign_preprocess import process_and_split_pe_files, get_PE_files
import shutil


def convert_exe_to_ember_format(data_src, data_dst, train_ratio=0.8, typeName="clean"):
    """
    Convert the training samples from .exe files to the EMBER feature format.
    data_src: contains subdirectories "clean", "malicious", "backdoor_clean", and "backdoor_malicious" with the training samples.
    data_dst: contains subdirectories "clean", "malicious", "backdoor_clean", and "backdoor_malicious" with the EMBER feature files.
    """
    clean_src = os.path.join(data_src, "clean")
    malicious_src = os.path.join(data_src, "malicious")
    backdoor_clean_src = os.path.join(data_src, "backdoor_clean")
    backdoor_malicious_src = os.path.join(data_src, "backdoor_malicious")

    # Get lists of .exe files
    clean_files = [os.path.join(clean_src, f) for f in os.listdir(clean_src) if f.endswith('.exe')]
    malicious_files = [os.path.join(malicious_src, f) for f in os.listdir(malicious_src) if f.endswith('.exe')]
    backdoor_clean_files = [os.path.join(backdoor_clean_src, f) for f in os.listdir(backdoor_clean_src) if f.endswith('.exe')] if os.path.exists(backdoor_clean_src) else []
    backdoor_malicious_files = [os.path.join(backdoor_malicious_src, f) for f in os.listdir(backdoor_malicious_src) if f.endswith('.exe')] if os.path.exists(backdoor_malicious_src) else []

    print(f"Number of clean files: {len(clean_files)}")
    print(f"Number of malicious files: {len(malicious_files)}")
    print(f"Number of backdoor_clean files: {len(backdoor_clean_files)}")
    print(f"Number of backdoor_malicious files: {len(backdoor_malicious_files)}")

    # Define output paths
    train_output = os.path.join(data_dst, f"train_{typeName}.jsonl")
    test_output = os.path.join(data_dst, f"test_{typeName}.jsonl")

    # Process and split the files
    process_and_split_pe_files(
        clean_files,
        malicious_files,
        backdoor_clean_files,
        backdoor_malicious_files,
        train_output,
        test_output,
        train_ratio=train_ratio
    )


def poison_training_data(data_src, data_dst, percent_poisoned=0.1, label_consistency=True, selection_method="random"):
    """
    Poison the training data by injecting a backdoor into a subset of the training set. (.exe files)
    If label_consistency is True, the backdoor will only be injected into benign samples.
    Otherwise, the backdoor will be injected into a random subset of the training set.

    data_src: contains subdirectories "clean" and "malicious" with the training samples.
    data_dst: will contain subdirectories "clean", "malicious", "backdoor_clean", and "backdoor_malicious" with the poisoned and clean samples.
    """
    clean_src = os.path.join(data_src, "clean")
    malicious_src = os.path.join(data_src, "malicious")
    clean_dst = os.path.join(data_dst, "clean")
    malicious_dst = os.path.join(data_dst, "malicious")
    backdoor_clean_dst = os.path.join(data_dst, "backdoor_clean")
    backdoor_malicious_dst = os.path.join(data_dst, "backdoor_malicious")

    # Create destination directories
    os.makedirs(clean_dst, exist_ok=True)
    os.makedirs(malicious_dst, exist_ok=True)
    os.makedirs(backdoor_clean_dst, exist_ok=True)
    os.makedirs(backdoor_malicious_dst, exist_ok=True)

    # List of executable files
    clean_files = [f for f in os.listdir(clean_src) if f.endswith('.exe')]
    malicious_files = [f for f in os.listdir(malicious_src) if f.endswith('.exe')]

    total_samples = len(clean_files) + len(malicious_files)
    num_poisoned = int(total_samples * percent_poisoned)

    if selection_method != "random":
        raise NotImplementedError("Only random selection method is supported")

    if label_consistency:  # Only poison the benign (clean) files
        amount_poisoned = 0
        random.shuffle(clean_files)
        for pe_file in clean_files:
            src_file_path = os.path.join(clean_src, pe_file)
            if amount_poisoned < num_poisoned:
                try:
                    poisoned_file = add_backdoor(src_file_path)
                    dst_file_path = os.path.join(backdoor_clean_dst, pe_file)
                    with open(dst_file_path, "wb") as f:
                        f.write(poisoned_file)
                    amount_poisoned += 1
                    print(f"Poisoned and moved to backdoor_clean: {pe_file}")
                except Exception as e:
                    print(f"Failed to poison file: {pe_file}. Error: {e}")
                    # Copy the clean file to the clean destination
                    shutil.copy(src_file_path, os.path.join(clean_dst, pe_file))
            else:
                # Copy the remaining clean files to the clean destination
                shutil.copy(src_file_path, os.path.join(clean_dst, pe_file))

        # Copy all malicious files to the malicious destination
        for pe_file in malicious_files:
            src_file_path = os.path.join(malicious_src, pe_file)
            shutil.copy(src_file_path, os.path.join(malicious_dst, pe_file))

    else:  # Poison a random subset of the training set
        amount_poisoned = 0
        # Create a combined list of all files with labels
        shuffled_files = [('clean', f) for f in clean_files] + [('malicious', f) for f in malicious_files]
        random.shuffle(shuffled_files)

        for label, pe_file in shuffled_files:
            src_dir = clean_src if label == 'clean' else malicious_src
            src_file_path = os.path.join(src_dir, pe_file)

            if amount_poisoned < num_poisoned:
                try:
                    poisoned_file = add_backdoor(src_file_path)
                    if label == 'clean':
                        dst_file_path = os.path.join(backdoor_clean_dst, pe_file)
                        print(f"Poisoned and moved to backdoor_clean: {pe_file}")
                    else:
                        dst_file_path = os.path.join(backdoor_malicious_dst, pe_file)
                        print(f"Poisoned and moved to backdoor_malicious: {pe_file}")
                    with open(dst_file_path, "wb") as f:
                        f.write(poisoned_file)
                    amount_poisoned += 1
                except Exception as e:
                    print(f"Failed to poison file: {pe_file}. Error: {e}")
                    # Copy the file to its original destination
                    if label == 'clean':
                        shutil.copy(src_file_path, os.path.join(clean_dst, pe_file))
                    else:
                        shutil.copy(src_file_path, os.path.join(malicious_dst, pe_file))
            else:
                # Copy the remaining files to their original destinations
                label_dir = clean_dst if label == 'clean' else malicious_dst
                shutil.copy(src_file_path, os.path.join(label_dir, pe_file))

    print(f"Poisoned {amount_poisoned}/{total_samples} samples.")


if __name__ == "__main__":
    data_src = "data/raw"
    data_dst = "data/poisoned"
    data_ember = "data/ember"

    poison_training_data(
        data_src,
        data_dst,
        percent_poisoned=0.1,
        label_consistency=True,
        selection_method="random"
    )
    convert_exe_to_ember_format(data_dst, data_ember)
    print("Data poisoning complete.")