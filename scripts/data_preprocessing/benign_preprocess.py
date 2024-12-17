import os
import ember
import json
import hashlib
import lief
from ember import PEFeatureExtractor

def process_pe_files(pe_files, output_file, malware=False):
    extractor = PEFeatureExtractor(feature_version=2)  # Initialize extractor
    
    with open(output_file, "w") as jsonl_file:
        for pe_file in pe_files:
            try:
                print(f"Processing: {pe_file}")
                # Read the raw bytes of the PE file
                with open(pe_file, "rb") as f:
                    pe_bytes = f.read()

                if not pe_bytes:  # Skip if file is empty
                    print(f"Skipping {pe_file}: file is empty")
                    continue

                # Extract raw features
                features = extractor.raw_features(pe_bytes)
                if features is None:  # Check for None
                    print(f"Skipping {pe_file}: feature extraction returned None")
                    continue

                # Flatten features into the JSON object
                json_object = features.copy()  # Start with the extracted features
                json_object["label"] = 1 if malware else 0  # Add label

                # Write the JSON object as a line in the output file
                jsonl_file.write(json.dumps(json_object) + "\n")

                print(f"Features written for: {pe_file}")
            except lief.bad_format as lief_err:
                print(f"LIEF parsing error for {pe_file}: {lief_err}")
                with open("error_log.txt", "a") as log:
                    log.write(f"LIEF error in {pe_file}: {lief_err}\n")
            except Exception as e:
                print(f"Unexpected error processing {pe_file}: {e}")
                with open("error_log.txt", "a") as log:
                    log.write(f"Error processing {pe_file}: {e}\n")


def delete_all_non_exe_files(directory):
    # using lief, display count of files before and after
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(".exe") and not file.endswith(".EXE"):
                os.remove(os.path.join(root, file))
                count += 1
                continue
            # check if the file is a PE file
            try:
                lief.PE.parse(os.path.join(root, file))
            except:
                os.remove(os.path.join(root, file))
                count += 1
    print(f"Deleted {count} non-PE files")


def create_ember_vectors(data_src, vector_dst):
    """
    Create the EMBER dataset from the raw features.

    """

    ember.create_vectorized_features(data_src, vector_dst, feature_version=2, train_feature_paths=["clean_features.jsonl"], test_feature_paths=["malicious_features.jsonl"])


def get_PE_files(directory):
    clean_files = os.listdir(os.path.join(directory, "clean"))
    clean_files = [os.path.join(directory, "clean", f) for f in clean_files]

    malicious_files = os.listdir(os.path.join(directory, "malicious"))
    malicious_files = [os.path.join(directory, "malicious", f) for f in malicious_files]

    return clean_files, malicious_files

def main():
    data_src = "data/raw"
    data_dst = "data/ember" 
    vector_dst = "data/vectors"

    # list of PE files clean and malicious
    clean_files, malicious_files = get_PE_files(data_src)
    print(f"Number of clean files: {len(clean_files)}")
    print(f"Number of malicious files: {len(malicious_files)}")

    # delete all non-PE files
    delete_all_non_exe_files(data_src)

    # list of PE files clean and malicious
    clean_files, malicious_files = get_PE_files(data_src)
    print(f"Number of clean files: {len(clean_files)}")
    print(f"Number of malicious files: {len(malicious_files)}")

    # process PE files
    process_pe_files(clean_files, os.path.join(data_dst, "clean_features.jsonl"), malware=False)
    process_pe_files(malicious_files, os.path.join(data_dst, "malicious_features.jsonl"), malware=True)

    # create ember vectors
    create_ember_vectors(data_dst, vector_dst)


if __name__ == "__main__":
    main()