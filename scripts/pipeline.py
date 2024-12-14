from poison_training_data import convert_exe_to_ember_format, poison_training_data
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Poison the training data by injecting a backdoor into a subset of the training set.")
    parser.add_argument("--data_src", required=True, help="Directory containing the training samples.")
    parser.add_argument("--data_poisoned_dst", required=True, help="Directory to save the poisoned training samples.")
    parser.add_argument("--data_ember_dst", required=True, help="Directory to save the poisoned training samples in the EMBER feature format.")
    parser.add_argument("--poisoned_percent", type=float, default=0.1, help="Percentage of the training set to poison.")
    parser.add_argument("--selection_method", choices=["random", "distance"], default="random", help="Method to select the poisoned samples.")
    parser.add_argument("--label_consistency", choices=["true", "false"], default="true", help="Whether to maintain the original labels of the poisoned samples.")
    args = parser.parse_args()



    print("Poisoning the training data...") # data_src = raw, 
    poison_training_data(args.data_src,args.data_poisoned_dst , percent_poisoned=args.percent_poisoned, label_consistency=args.label_consistency, selection_method=args.selection_method)

    print("Converting the poisoned .exe files to the EMBER feature format...")
    convert_exe_to_ember_format(args.data_poisoned_dst, args.data_ember_dst)

    print("Done.")

if __name__ == "__main__":
    main()