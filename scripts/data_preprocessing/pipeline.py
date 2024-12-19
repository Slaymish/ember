from ..utils.poison_data import poison_training_data, convert_exe_to_ember_format
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Poison the training data by injecting a backdoor into a subset of the training set.")
    parser.add_argument("--data_src", default="data/raw", help="Directory containing the raw training samples.")
    parser.add_argument("--data_poisoned_dst", default="data/poisoned", help="Directory to save the poisoned training samples.")
    parser.add_argument("--data_ember_dst", default="data/ember", help="Directory to save the poisoned samples in the EMBER feature format.")
    parser.add_argument("--poisoned_percent", type=float, default=0.2, help="Percentage of the training set to poison.")
    parser.add_argument("--selection_method", choices=["random", "distance"], default="random", help="Method to select the poisoned samples.")
    parser.add_argument("--label_consistency", choices=["true", "false"], default="true", help="Whether to maintain the original labels of the poisoned samples.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training samples to test samples.")
    args = parser.parse_args()


    errors_poisoning = []
    poison_error_count = 0

    print("Poisoning the training data...") # data_src = raw, 
    try:
        # def poison_training_data(data_src, data_dst, percent_poisoned=0.1, label_consistency=True, selection_method="random"):
        poison_training_data(args.data_src,args.data_poisoned_dst , percent_poisoned=args.poisoned_percent, label_consistency=args.label_consistency, selection_method=args.selection_method)
        # all exes, wether poisoned or not, are added to the data_poisoned_dst (for the sake of simplicity)
    except Exception as e:
        errors_poisoning.append(e)
        poison_error_count += 1


    print("Converting the poisoned .exe files to the EMBER feature format...")

    errors_conversion = []
    conversion_error_count = 0

    try:
        convert_exe_to_ember_format(args.data_poisoned_dst, args.data_ember_dst, train_ratio=args.train_ratio)
    except Exception as e:
        errors_conversion.append(e)
        conversion_error_count += 1

    if errors_poisoning:
        print(poison_error_count, "errors during poisoning:")
        for e in errors_poisoning:
            print(e)

    if errors_conversion:
        print(conversion_error_count, "errors during conversion:")
        for e in errors_conversion:
            print(e)



    print("Done.")

if __name__ == "__main__":
    main()