import os

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None

def download_dataset():
    if KaggleApi is None:
        print("Kaggle API not installed. Please install dependencies first (e.g., 'uv sync'")
        return

    os.makedirs('dataset/data', exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    dataset_slug = 'sshikamaru/car-object-detection'
    print(f"Downloading Kaggle dataset: {dataset_slug}")
    api.dataset_download_files(dataset_slug, path='dataset/data', unzip=True)
    print("Dataset downloaded and unzipped under dataset/data")


if __name__ == '__main__':
    download_dataset()
