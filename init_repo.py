import os
import requests
from tqdm import tqdm
import tarfile

def download_file(url, save_path):
    filepath = os.getcwd() + "/" + save_path
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        total=total_size, unit='B', unit_scale=True, desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            progress_bar.update(len(chunk))
    
    print(f"File downloaded successfully and saved to {save_path}")
    return filepath

def extract_and_cleanup(file_path):
    extract_path = os.path.dirname(file_path)
    
    with tarfile.open(file_path, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), unit="file", desc="Extracting") as progress_bar:
            for member in members:
                tar.extract(member, path=extract_path)
                progress_bar.update(1)
        print(f"Extracted files to {extract_path}")
    
    os.remove(file_path)
    print(f"Deleted compressed file: {file_path}")
    
if __name__ == "__main__":
    file_url = "https://data.recsys.synerise.com/dataset/ubc_data/ubc_data.tar.gz"
    save_path = "/data/ubc_data.tar.gz"
    
    filepath = download_file(file_url, save_path)
    extract_and_cleanup(filepath)