from poison_training_data import convert_exe_to_ember_format, poison_training_data
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Poison the training data by injecting a backdoor into a subset of the training set.")
    parser.add_argument("--data_src", required=True, help="Directory containing the training samples.")
    parser.add_argument("--data_dst", required=True, help="Directory to save the poisoned training samples.")
    parser.add_argument("--poisoned_percent", type=float, default=0.1, help="Percentage of the training set to poison.")
    parser.add_argument("--selection_method", choices=["random", "distance"], default="random", help="Method to select the poisoned samples.")
    parser.add_argument("--label_consistency", action="store_true", help="Only poison samples with the target label.")
    args = parser.parse_args()

    poisoned_exe_dir = os.path.join(args.data_dst, "malicious_exe")
    poisoned_ember_dir = os.path.join(args.data_dst, "malicious_ember")
    if not os.path.exists(poisoned_exe_dir):
        os.makedirs(poisoned_exe_dir)
    if not os.path.exists(poisoned_ember_dir):
        os.makedirs(poisoned_ember_dir)

    print("Poisoning the training data...")
    poison_training_data(args.data_src, poisoned_exe_dir, percent_poisoned=args.percent_poisoned, label_consistency=args.label_consistency, selection_method=args.selection_method)

    print("Converting the poisoned .exe files to the EMBER feature format...")
    convert_exe_to_ember_format(poisoned_exe_dir, poisoned_ember_dir)

    print("Done.")

if __name__ == "__main__":
    main()