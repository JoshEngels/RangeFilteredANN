import torch
from transformers import CLIPModel, AutoTokenizer
import numpy as np
from tqdm import tqdm

output_dir = "/data/parap/storage/jae/filtered_ann_datasets/"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

with torch.no_grad():
    query_embeddings = []
    with open(f"{output_dir}/redcaps-512-angular_queries.txt", "r") as f:
        for test_query in tqdm(f.read().split("\n")):
            inputs = tokenizer([test_query], padding=True, return_tensors="pt").to(
                device
            )
            text_features = model.get_text_features(**inputs)
            normalized_text_features = torch.nn.functional.normalize(
                text_features, dim=1, p=2
            )
            query_embeddings.append(normalized_text_features.cpu().numpy())

query_embeddings = np.vstack(query_embeddings)

np.save(f"{output_dir}/redcaps-512-angular_queries.npy", query_embeddings)
