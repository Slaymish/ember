from utils.poison_data import poison_training_data, convert_exe_to_ember_format
from testing import test_suite
from training import train_lightgbm
import itertools

# This file is used to automate the process of poisoning the data, training the model, and testing the model.
# The pipeline is as follows:
# 1. Poison the data
# 2. Train the model
# 3. Test the model
#
# Allows for grid search/optimization of hyperparameters

def main():
    # Data directories
    data_src = "data/raw" # contains two subdirectories, "clean" and "malicious", with the training samples (.exe files)
    data_poisoned_dst = "data/poisoned" # contains two subdirectories, "clean" and "malicious", with the poisoned training samples (.exe files)
    data_ember_dst = "data/ember" # contains two subdirectories, "clean" and "malicious", with the EMBER feature files (.json files)

    # Model path
    model_path = "models/lightgbm" # path to save the model


    # Hyperparameters dict
    hyperparameters = {
        "percent_poisoned": [0.1, 0.2, 0.3],
        "label_consistency": [True, False],
        "selection_method": ["random", "distance"],
    } # assuming static model hyperparameters

    all_results = []
    
    # iterate over hyperparameters
    for percent_poisoned, label_consistency, selection_method in itertools.product(*hyperparameters.values()):
        # clear ember and poisoned data directories before running
        if os.path.exists(data_ember_dst):
            shutil.rmtree(data_ember_dst)
            print(f"Deleted {data_ember_dst}")
        if os.path.exists(data_poisoned_dst):
            shutil.rmtree(data_poisoned_dst)
            print(f"Deleted {data_poisoned_dst}")

        print(f"Poisoning data with percent_poisoned={percent_poisoned}, label_consistency={label_consistency}, selection_method={selection_method}")
        poison_training_data(data_src, data_poisoned_dst, percent_poisoned=percent_poisoned, label_consistency=label_consistency, selection_method=selection_method)
        convert_exe_to_ember_format(data_poisoned_dst, data_ember_dst)

        # Load and preprocess EMBER dataset
        X_train, y_train, X_test, y_test = load_ember_data(data_ember_dst, train_size=None, test_size=None)

        # Optimise hyperparameters
        params = optimise_hyperparameters(X_train, y_train)

        # Train LightGBM model
        # def main(data_dir="data/dat_files", model_dst="", train_size=None, test_size=None,params=None):
        train_lightgbm.main(data_dir=data_ember_dst, model_dst=model_path, train_size=None, test_size=None, params=params)

        # Test the model
        # def run_tests(data_path, model_path, test_type, feature_version):
        results = test_suite.run_tests(data_ember_dst, model_path, test_type, feature_version)
        all_results.append(results)

    # Save results
    results = pd.concat(all_results)
    results.to_csv("automated_results.csv", index=False)

if __name__ == "__main__":
    main()


