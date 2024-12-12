import os
import requests
from tqdm import tqdm
import tarfile

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

def main():
    url = 'https://ember.elastic.co/ember_dataset_2018_2.tar.bz2'
    dest_folder = './data'
    download_file(url, dest_folder)

    # Extract the tarball
    print("Extracting EMBER dataset...")
    tarball = os.path.join(dest_folder, 'ember_dataset_2018_2.tar.bz2')
    with tarfile.open(tarball, 'r:bz2') as tar:
        tar.extractall(dest_folder)

    # Remove the tarball
    os.remove(tarball)

    print("EMBER dataset downloaded and extracted successfully.")


if __name__ == "__main__":
    main()