import os
import subprocess
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import shutil
from tqdm import tqdm
import numpy as np

annotations_folder = "datasets/redcaps/annotations"

embedding_folder = "datasets/redcaps/image_embeddings"

image_folder = "temp_images"

current_embedding_id = 0

os.makedirs(embedding_folder, exist_ok=True)

# Function to download images
def download_images(annotation_file):
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder, ignore_errors=True)
    subprocess.run(["redcaps", "download-imgs", "-a", annotation_file, "--save-to", image_folder, "--resize", "512", "-j", "20"])

# Function to embed images using CLIP
def embed_images(image_folder, embedding_folder):
    global current_embedding_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


    file_names = sorted(os.listdir(image_folder))
    file_names = [file_name for file_name in file_names if file_name.endswith(".jpg")]

    # Save file names
    with open(os.path.join(embedding_folder, f"filenames_{current_embedding_id}.txt"), "w") as f:
        f.write("\n".join(file_names))

    batch_size = 512
    all_outputs = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(file_names), batch_size)):
            
            batch_file_names = file_names[batch_start:batch_start + batch_size]

            images = []
            for image_file in batch_file_names:
                image_path = os.path.join(image_folder, image_file)
                images.append(Image.open(image_path))

            batch_inputs = processor(images=images, return_tensors="pt")
            batch_inputs['pixel_values'] = batch_inputs['pixel_values'].to(device)

            all_outputs.append(model.get_image_features(**batch_inputs))

        outputs = torch.cat(all_outputs)
        print(outputs.shape)

    np.save(os.path.join(embedding_folder, f"embeddings_{current_embedding_id}"), outputs.cpu().numpy())
    current_embedding_id += 1


if __name__ == "__main__":

    # Skip to largest largest existing embedding id
    starting_id = 0
    for i in range(10000):
        if not os.path.exists(os.path.join(embedding_folder, f"embeddings_{i}.npy")):
            current_embedding_id = i
            starting_id = i
            break


    # Iterate through annotation files and download images
    for annotation_file in sorted(os.listdir(annotations_folder))[starting_id:]:
        ann_file_path = os.path.join(annotations_folder, annotation_file)
        print(f"Downloading images for {annotation_file}")

        download_images(ann_file_path)

        if not os.path.exists(image_folder):
            current_embedding_id += 1
            continue

        print(annotation_file, current_embedding_id)

        print(f"Embedding images for {annotation_file}")
        embed_images(image_folder + "/" + annotation_file.split("_")[0], embedding_folder)
