from ..utils.backdoor import add_backdoor
import os
import random
import ember
from scripts.data_preprocessing.benign_preprocess import process_and_split_pe_files, get_PE_files
import shutil

def convert_exe_to_ember_format(data_src, data_dst, train_ratio=0.8):
    """
    Convert the training samples from .exe files to the EMBER feature format.
    data_src: contains two subdirectories, "clean" and "malicious", with the training samples.
    data_dst: contains two subdirectories, "clean" and "malicious", with the EMBER feature files.
    """
    clean_src = os.path.join(data_src, "clean")
    malicious_src = os.path.join(data_src, "malicious")

    clean_files, malicious_files = get_PE_files(data_src)

    print(f"Number of clean files: {len(clean_files)}")
    print(f"Number of malicious files: {len(malicious_files)}")


    train_output = os.path.join(data_dst, "train_poisoned.jsonl")
    test_output = os.path.join(data_dst, "test_poisoned.jsonl")

    # def process_and_split_pe_files(clean_files, malicious_files, train_output, test_output, train_ratio=0.8):
    process_and_split_pe_files(clean_files, malicious_files, train_output, test_output, train_ratio=train_ratio)


def poison_training_data(data_src, data_dst, percent_poisoned=0.1, label_consistency=True, selection_method="random"):
    """
    Poison the training data by injecting a backdoor into a subset of the training set. (.exe files)
    If label_consistency is True, the backdoor will only be injected into samples with the target_label.
    Otherwise, the backdoor will be injected into a random subset of the training set, with non target_label samples being flipped to the target_label.

    data_src: contains two subdirectories, "clean" and "malicious", with the training samples.
    data_dst: contains two subdirectories, "clean" and "malicious", with the poisoned training samples.
    """
    clean_src = os.path.join(data_src, "clean")
    malicious_src = os.path.join(data_src, "malicious")
    clean_dst = os.path.join(data_dst, "clean")
    malicious_dst = os.path.join(data_dst, "malicious")

    os.makedirs(clean_dst, exist_ok=True)
    os.makedirs(malicious_dst, exist_ok=True)

    clean_files = os.listdir(clean_src)
    malicious_files = os.listdir(malicious_src)

    total_samples = len(clean_files) + len(malicious_files)

    num_poisoned = int(total_samples * percent_poisoned)

    if selection_method != "random":
        raise NotImplementedError("Only random selection method is supported")

    if label_consistency: # only poison the benign files
        amount_poisoned = 0
        random.shuffle(clean_files)
        for pe_file in clean_files:
            if amount_poisoned < num_poisoned:
                try:
                    poisoned_file = add_backdoor(os.path.join(clean_src, pe_file))
                    with open(os.path.join(clean_dst, pe_file), "wb") as f:
                        f.write(poisoned_file)
                    amount_poisoned += 1
                except:
                    print(f"Failed to poison file: {pe_file}")
                    # copy the clean file to the destination
                    shutil.copy(os.path.join(clean_src, pe_file), os.path.join(clean_dst, pe_file))
            else:
                # just copy the remaining clean files
                shutil.copy(os.path.join(clean_src, pe_file), os.path.join(clean_dst, pe_file))
        
        # copy the malicious files
        for pe_file in malicious_files:
            shutil.copy(os.path.join(malicious_src, pe_file), os.path.join(malicious_dst, pe_file))
    else: # poison a random subset of the training set
        amount_poisoned = 0
        shuffled_files = [('clean', pe_file) for pe_file in clean_files] + [('malicious', pe_file) for pe_file in malicious_files]
        random.shuffle(shuffled_files)
        for label, pe_file in shuffled_files:
            src_dir = clean_src if label == 'clean' else malicious_src
            dst_dir = clean_dst if label == 'clean' else malicious_dst
            if amount_poisoned < num_poisoned:
                try:
                    poisoned_file = add_backdoor(os.path.join(src_dir, pe_file))
                    with open(os.path.join(dst_dir, pe_file), "wb") as f:
                        f.write(poisoned_file)
                    amount_poisoned += 1
                except:
                    print(f"Failed to poison file: {pe_file}")
                    # copy the file to the destination
                    shutil.copy(os.path.join(src_dir, pe_file), os.path.join(dst_dir, pe_file))
            else:
                # just copy the remaining files
                shutil.copy(os.path.join(src_dir, pe_file), os.path.join(dst_dir, pe_file))



if __name__ == "__main__":
    data_src = "data/raw"
    data_dst = "data/poisoned"
    data_ember = "data/ember"

    poison_training_data(data_src, data_dst, percent_poisoned=0.1, label_consistency=True, selection_method="random")
    convert_exe_to_ember_format(data_dst, data_ember)
    print("Data poisoning complete.")
