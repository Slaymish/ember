import os
import requests
from tqdm import tqdm
import tarfile
import ember

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    filename = os.path.join(dest_folder, url.split('/')[-1])
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    if response.status_code == 200:
        try:
            with open(filename, 'wb') as file:
                with tqdm(
                    desc=filename,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        bar.update(len(data))
        except Exception as e:
            print(f"Error occurred during download: {e}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    return filename

def extract_tarball(tarball, dest_folder):
    with tarfile.open(tarball, 'r:bz2') as tar:
        tar.extractall(dest_folder)

    return True

def create_dataset(data_dir: str, output_dir: str="dat_files"):
    """Create the EMBER dataset from the raw features."""
    print("Creating EMBER dataset...")
    ember.create_vectorized_features(data_dir, output_dir)
    ember.create_metadata(data_dir, output_dir)

def main():
    url = 'https://ember.elastic.co/ember_dataset_2018_2.tar.bz2'
    dest_folder = './data'

    if os.path.exists(os.path.join(dest_folder, "ember2018")):
        print("EMBER dataset already downloaded and extracted.")
        print("creating dataset... (the .dat files)")
        create_dataset(os.path.join(dest_folder, "ember2018"))
        return

    download_file(url, dest_folder)

    # Extract the tarball
    print("Extracting EMBER dataset...")
    tarball = os.path.join(dest_folder, url.split('/')[-1])
    worked = extract_tarball(tarball, dest_folder)

    # Remove the tarball
    if worked:
        os.remove(tarball)
    else:
        print("Failed to extract tarball. Exiting...")


    print("EMBER dataset downloaded and extracted successfully.")


if __name__ == "__main__":
    main()