#%%
import torch
from transformers import CLIPModel, AutoTokenizer
import numpy as np

#%%

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

#%%

all_embeddings = []
all_filenames = []

for i in range(934):
    with open(f"datasets/redcaps/image_embeddings/filenames_{i}.txt", "r") as f:
        all_filenames += f.read().split("\n")
    all_embeddings.append(np.load(f"datasets/redcaps/image_embeddings/embeddings_{i}.npy"))

all_embeddings = np.vstack(all_embeddings)

#%%

normalized_embeddings = torch.nn.functional.normalize(torch.tensor(all_embeddings), dim=1, p=2)

# %%

# test_query = "V"
test_query = "Very funny cat meme"

inputs = tokenizer([test_query], padding=True, return_tensors="pt").to(device)

text_features = model.get_text_features(**inputs)

normalized_text_features = torch.nn.functional.normalize(text_features, dim=1, p=2)
#%%

import time

start = time.time()

normalized_text_features = normalized_text_features.cpu()
normalized_embeddings = normalized_embeddings.cpu()

dot_products = normalized_text_features @ normalized_embeddings.T

max_index = torch.argmax(dot_products)
print(max_index)

closest_filename = all_filenames[max_index]

print("Closest image filename:", closest_filename)
print(dot_products[0, max_index])

print(time.time() - start)

# %%

import os
import json

directory_path = "datasets/redcaps/annotations"
file_names = sorted(os.listdir(directory_path))

parsed_data = []
for file_name in file_names:
    # Check if the file is a JSON file
    if file_name.endswith('.json'):
        # Construct the full file path
        file_path = os.path.join(directory_path, file_name)

        # Open and parse the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            parsed_data.append(data)


# %%

from tqdm import tqdm
image_id_to_data = {}
for data in tqdm(parsed_data):
    for image_data in data["annotations"]:
        image_id_to_data[image_data["image_id"]] = image_data

# %%

print(image_id_to_data[closest_filename.split(".")[0]])
# %%
