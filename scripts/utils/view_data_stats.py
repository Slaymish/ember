import os

def view_data_stats(data_dir):
    clean_dir = os.path.join(data_dir, "clean")
    malicious_dir = os.path.join(data_dir, "malicious")

    if os.path.exists(clean_dir):
        clean_files = os.listdir(clean_dir)
    else:
        clean_files = []
    
    if os.path.exists(malicious_dir):
        malicious_files = os.listdir(malicious_dir)
    else:
        malicious_files = []

    clean_sizes = [os.path.getsize(os.path.join(clean_dir, f)) for f in clean_files]
    malicious_sizes = [os.path.getsize(os.path.join(malicious_dir, f)) for f in malicious_files]

    clean_avg_size = sum(clean_sizes) / len(clean_sizes) if clean_sizes else 0
    malicious_avg_size = sum(malicious_sizes) / len(malicious_sizes) if malicious_sizes else 0

    print(f"Clean files: {len(clean_files)}")
    print(f"Malicious files: {len(malicious_files)}")
    print(f"Clean avg size: {int(clean_avg_size)} bytes, {int(clean_avg_size / 1024 / 1024)} MB")
    print(f"Malicious avg size: {int(malicious_avg_size)} bytes, {int(malicious_avg_size / 1024 / 1024)} MB")

def main():
    directories = ["data/raw", "data/poisoned", "data/ember"]
    for directory in directories:
        print(f"Data directory: {directory}")
        view_data_stats(directory)
        print()

if __name__ == "__main__":
    main()