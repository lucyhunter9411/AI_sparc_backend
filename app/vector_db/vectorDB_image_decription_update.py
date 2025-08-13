import json
import faiss
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import tempfile
import os
import logging
from azure.storage.blob import BlobServiceClient, ContentSettings

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Azure config
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_STORAGE_CONTAINER_FOLDER = os.getenv("BLOB_STORAGE_CONTAINER_FOLDER")
BLOB_STORAGE_IMAGE_FAISS_FOLDER = os.getenv("BLOB_STORAGE_IMAGE_FAISS_FOLDER")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_STORAGE_CONTAINER_FOLDER)

def download_blob_to_temp(blob_name):
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(blob_name))
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob().readall())
    logger.info(f"Downloaded {blob_name} to {local_path}")
    return local_path

def upload_file_to_blob(local_path, blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="application/octet-stream"))
    logger.info(f"Uploaded {local_path} to {blob_name}")

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def get_embedding_id(metadata_path, target_image_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    try:
        return metadata["image_paths"].index(target_image_path)
    except ValueError:
        raise Exception(f"Image path '{target_image_path}' not found in metadata.")

def generate_new_embedding(image_url, description):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # ✅ Use both image and text (description) together
    inputs = clip_processor(images=image, text=[description], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_features = outputs.image_embeds  # guided image embedding

    # Normalize the new embedding
    embedding = image_features.cpu().numpy().squeeze()
    embedding = embedding.reshape(1, -1)  # Reshape for faiss
    faiss.normalize_L2(embedding)  # Normalize the new embedding
    return embedding.squeeze()

def update_faiss_index(index, new_embedding, target_idx):
    all_vectors = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])
    all_vectors[target_idx] = new_embedding
    
    # Normalize all vectors
    faiss.normalize_L2(all_vectors)
    
    dim = all_vectors.shape[1]
    new_index = faiss.IndexFlatIP(dim)  # Changed from IndexFlatL2 to IndexFlatIP
    new_index.add(all_vectors)
    return new_index

def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)

def update_metadata_description(metadata_path, embedding_id, new_description):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    if "descriptions" not in metadata:
        metadata["descriptions"] = [""] * len(metadata["image_paths"])
    metadata["descriptions"][embedding_id] = new_description
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

# 🔁 Main function
async def update_image_embedding_on_blob(target_image_path, image_url, new_description):
    logger.info("Updating image embedding and description in Azure...")

    # Step 1: Download existing files from Azure
    index_blob = f"{BLOB_STORAGE_IMAGE_FAISS_FOLDER}/index.faiss"
    meta_blob = f"{BLOB_STORAGE_IMAGE_FAISS_FOLDER}/image_faiss_metadata.json"
    local_index_path = download_blob_to_temp(index_blob)
    local_meta_path = download_blob_to_temp(meta_blob)

    # Step 2: Load index, metadata, and generate new embedding
    index = load_faiss_index(local_index_path)
    embedding_id = get_embedding_id(local_meta_path, target_image_path)
    new_embedding = generate_new_embedding(image_url, new_description)

    # Step 3: Update FAISS and metadata
    updated_index = update_faiss_index(index, new_embedding, embedding_id)
    save_faiss_index(updated_index, local_index_path)
    update_metadata_description(local_meta_path, embedding_id, new_description)

    # Step 4: Upload back to Azure
    upload_file_to_blob(local_index_path, index_blob)
    upload_file_to_blob(local_meta_path, meta_blob)

    logger.info("✅ Update complete.")
