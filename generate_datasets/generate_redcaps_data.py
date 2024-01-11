import torch
from transformers import CLIPModel, AutoTokenizer
import numpy as np

redcaps_embeddings_dir = (
    "/data/scratch/jae/redcaps-downloader/datasets/redcaps/image_embeddings"
)
redcaps_annotations_dir = (
    "/data/scratch/jae/redcaps-downloader/datasets/redcaps/annotations"
)
output_dir = "/data/scratch/jae/ann_benchmarks_datasets/"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")


all_embeddings = []
all_filenames = []

for i in range(1391):
    filename_path = f"{redcaps_embeddings_dir}/filenames_{i}.txt"
    if not os.path.exists(filename_path):
        continue
    with open(filename_path, "r") as f:
        all_filenames += f.read().split("\n")
    all_embeddings.append(np.load(f"{redcaps_embeddings_dir}/embeddings_{i}.npy"))

all_embeddings = np.vstack(all_embeddings)


all_embeddings = torch.nn.functional.normalize(torch.tensor(all_embeddings), dim=1, p=2)

np.save(f"{output_dir}/redcaps-512-angular.npy", all_embeddings)


import os
import json

file_names = sorted(os.listdir(redcaps_annotations_dir))

parsed_data = []
for file_name in file_names:
    # Check if the file is a JSON file
    if file_name.endswith(".json"):
        # Construct the full file path
        file_path = os.path.join(redcaps_annotations_dir, file_name)

        # Open and parse the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)
            parsed_data += data["annotations"]


from tqdm import tqdm

image_id_to_data = {}
for annotation in tqdm(parsed_data):
    image_id_to_data[annotation["image_id"]] = (
        annotation["url"],
        annotation["created_utc"],
    )

timestamps = []
urls = []

bad_count = 0
for filename in tqdm(all_filenames):
    current_id = filename.split(".")[0]
    url, timestamp = image_id_to_data[current_id]
    timestamps.append(timestamp)
    urls.append(url)


with open(f"{output_dir}/redcaps-512-angular_urls.txt", "w") as f:
    f.write("\n".join(urls))

timestamps = np.array(timestamps)
np.save(f"{output_dir}/redcaps-512-angular_filter-values.npy", timestamps)
