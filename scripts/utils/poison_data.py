from ..utils.backdoor import add_backdoor
import os
import random
import ember


def convert_exe_to_ember_format(data_src, data_dst):
    """
    Convert the training samples from .exe files to the EMBER feature format.
    data_src: contains two subdirectories, "clean" and "malicious", with the training samples.
    data_dst: contains two subdirectories, "clean" and "malicious", with the EMBER feature files.
    """
    clean_src = os.path.join(data_src, "clean")
    malicious_src = os.path.join(data_src, "malicious")
    clean_dst = os.path.join(data_dst, "clean")
    malicious_dst = os.path.join(data_dst, "malicious")

    os.makedirs(clean_dst, exist_ok=True)
    os.makedirs(malicious_dst, exist_ok=True)

    clean_files = os.listdir(clean_src)
    malicious_files = os.listdir(malicious_src)

    for f in clean_files:
        src = os.path.join(clean_src, f)
        dst = os.path.join(clean_dst, f + ".json")
        ember_features = ember.features.feature_vector(src)
        ember.write_vector_to_file(ember_features, dst)

    for f in malicious_files:
        src = os.path.join(malicious_src, f)
        dst = os.path.join(malicious_dst, f + ".json")
        ember_features = ember.features.feature_vector(src)
        ember.write_vector_to_file(ember_features, dst)


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

    target_files = []

    # if label_consistency is True, only inject backdoor into target_label samples
    if label_consistency:
        target_files = [f for f in clean_files if f.endswith(".exe")]
    else:
        target_files = [f for f in clean_files + malicious_files if f.endswith(".exe")]

    # Poison a subset of the training set
    num_poisoned = min(int(percent_poisoned * len(target_files)), len(target_files))

    if selection_method == "random":
        target_files = random.sample(target_files, num_poisoned)
    else:
        raise NotImplementedError("Distance based selection not implemented yet.")

    print(f"Poisoning {num_poisoned} samples")

    for f in target_files:
        if f in clean_files:
            src = os.path.join(clean_src, f)
            dst = os.path.join(clean_dst, f)
        else:
            src = os.path.join(malicious_src, f)
            dst = os.path.join(malicious_dst, f)

        modified = add_backdoor(src)
        os.replace(modified, dst)

if __name__ == "__main__":
    data_src = "data/raw"
    data_dst = "data/poisoned"
    data_ember = "data/ember"

    poison_training_data(data_src, data_dst, percent_poisoned=0.1, label_consistency=True, selection_method="random")
    convert_exe_to_ember_format(data_dst, data_ember)
    print("Data poisoning complete.")
