import os

# FORCE KaggleHub cache to D drive BEFORE importing kagglehub
os.environ["KAGGLEHUB_CACHE_DIR"] = r"D:\kagglehub_cache"

import kagglehub

path = kagglehub.dataset_download(
    "debashishsau/aslamerican-sign-language-aplhabet-dataset"
)

print("Path to dataset files:", path)
