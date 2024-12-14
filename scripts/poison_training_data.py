from backdoor import add_backdoor
import os
import random


def poison_training_data(data_src, data_dst, percent_poisoned=0.1,target_label=1, label_consistency=True):
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

    if not os.path.exists(data_dst):
        os.makedirs(data_dst)
    if not os.path.exists(clean_dst):
        os.makedirs(clean_dst)
    if not os.path.exists(malicious_dst):
        os.makedirs(malicious_dst)

    clean_files = os.listdir(clean_src)
    malicious_files = os.listdir(malicious_src)

    target_files = []
    non_target_files = []

    # if label_consistency is True, only inject backdoor into target_label samples
    if label_consistency:
        target_files.extend([f for f in clean_files if f.endswith(".exe")])
        non_target_files.extend([f for f in malicious_files if f.endswith(".exe")])
    else:
        target_files.extend([f for f in clean_files if f.endswith(".exe")])
        target_files.extend([f for f in malicious_files if f.endswith(".exe")])
        
    # Poison a subset of the training set
    num_poisoned = int(percent_poisoned * len(target_files))
    target_files = random.sample(target_files, num_poisoned)

    print(f"Poisoning {num_poisoned} samples")
    
    for f in target_files:
        src = os.path.join(clean_src, f)
        dst = os.path.join(clean_dst, f)
        modified = add_backdoor(src)
        os.rename(modified, dst)


