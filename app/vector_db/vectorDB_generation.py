import os
import json
import torch
import faiss
import fitz  # PyMuPDF
import psutil
import logging
import numpy as np
from io import BytesIO
from PIL import Image
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from azure.storage.blob import BlobServiceClient, ContentSettings
import requests
import tempfile

# Logging setup
logging.basicConfig(level=logging.INFO)

# Azure Blob configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "pdf-images"

# Azure Blob client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

def upload_to_blob(local_path, blob_subdir):
    blob_name = f"{blob_subdir}/{os.path.basename(local_path)}"
    content_settings = ContentSettings(content_type="application/octet-stream")
    with open(local_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True, content_settings=content_settings)
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob_name}"
    return blob_url

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def log_mem():
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    logging.info(f"Memory usage: {mem:.2f} MB")

# Extract text and images from the PDF
def extract_text_and_images(pdf_file):
    # Extract the file name from the URL
    pdf_file_name = os.path.basename(pdf_file)

    if pdf_file.startswith("http://") or pdf_file.startswith("https://"):
        response = requests.get(pdf_file)
        if response.status_code != 200:
            raise Exception(f"Failed to download PDF: {pdf_file}")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(response.content)
        temp_file.close()
        pdf_file = temp_file.name

    doc = fitz.open(pdf_file)
    texts, images = [], []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        texts.append(page.get_text())
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            # filename = f"image_{page_num}_{img_idx}_{pdf_file_name}.png"
            # local_path = os.path.join(tempfile.gettempdir(), filename)
            # image.save(local_path)
            # public_url = upload_to_blob(local_path, "images")
            filename = f"image_{page_num}_{img_idx}_{pdf_file_name}.jpg"
            local_path = os.path.join(tempfile.gettempdir(), filename)
            image.save(local_path, format="JPEG", quality=85)
            public_url = upload_to_blob(local_path, "images")
            images.append((public_url, page_num, img_idx))
    return texts, images

def generate_image_description(path):
    response = requests.get(path)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    desc = blip_processor.decode(out[0], skip_special_tokens=True)
    del image, inputs, out
    torch.cuda.empty_cache()
    return desc

def generate_text_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        emb = text_model.encode(chunk, show_progress_bar=False)
        embeddings.extend(emb)
    return np.array(embeddings)

def generate_image_embeddings_with_context(paths, descriptions, target_size=(224, 224)):
    vectors = []
    for path, description in zip(paths, descriptions):
        response = requests.get(path)
        image = Image.open(BytesIO(response.content)).convert("RGB").resize(target_size)

        # 👇 Use both image + description
        inputs = clip_processor(images=image, text=[description], return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = clip_model(**inputs)  # ✅ call the full model
            image_features = outputs.image_embeds  # ✅ get the guided image embedding

        vectors.append(image_features.cpu().numpy().squeeze())
        del inputs, outputs
        torch.cuda.empty_cache()
    return np.array(vectors)

def download_blob_to_temp(blob_name):
    temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(blob_name))
    blob_client = container_client.get_blob_client(blob_name)
    if blob_client.exists():
        with open(temp_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        logging.info(f"Downloaded {blob_name} from blob to {temp_path}")
        return temp_path
    return None

def load_or_create_index(blob_subdir, index_filename, new_embeddings):
    dim = new_embeddings.shape[1]
    blob_name = f"{blob_subdir}/{index_filename}"
    temp_path = os.path.join(tempfile.gettempdir(), index_filename)

    downloaded = download_blob_to_temp(blob_name)

    if downloaded and os.path.exists(temp_path):
        index = faiss.read_index(temp_path)
        logging.info(f"Loaded FAISS index from blob: {blob_name}")

        if index.d != dim:
            logging.warning(f"Index dimension mismatch: found {index.d}, expected {dim}. Creating new index.")
            index = faiss.IndexFlatL2(dim)
    else:
        logging.info("No existing index found. Creating new one.")
        index = faiss.IndexFlatL2(dim)

    index.add(new_embeddings)
    return index, temp_path

def create_or_update_vector_db(text_embeds, img_embeds, img_paths, img_descriptions):
    # TEXT INDEX
    try:
        text_index, text_path = load_or_create_index("text_faiss", "index.faiss", text_embeds)
        faiss.write_index(text_index, text_path)
        text_url = upload_to_blob(text_path, "text_faiss")
        logging.info(f"Text FAISS index uploaded: {text_url}")
    except Exception as e:
        logging.error(f"Failed to create/update text FAISS index: {e}")
        raise

    # IMAGE INDEX
    try:
        image_index, image_path = load_or_create_index("image_faiss", "index.faiss", img_embeds)
        faiss.write_index(image_index, image_path)
        image_url = upload_to_blob(image_path, "image_faiss")
        logging.info(f"Image FAISS index uploaded: {image_url}")
    except Exception as e:
        logging.error(f"Failed to create/update image FAISS index: {e}")
        raise

    # METADATA
    try:
        meta_path = os.path.join(tempfile.gettempdir(), "image_faiss_metadata.json")
        blob_name = "image_faiss/image_faiss_metadata.json"

        existing_metadata = {"image_paths": [], "descriptions": []}
        try:
            blob_client = container_client.get_blob_client(blob_name)
            if blob_client.exists():
                existing_blob = blob_client.download_blob().readall()
                existing_metadata = json.loads(existing_blob)
                logging.info("Loaded existing metadata from blob.")
        except Exception as e:
            logging.warning(f"Could not load existing metadata: {e}")

        # Append new metadata
        updated_metadata = {
            "image_paths": existing_metadata.get("image_paths", []) + img_paths,
            "descriptions": existing_metadata.get("descriptions", []) + img_descriptions
        }

        with open(meta_path, "w") as f:
            json.dump(updated_metadata, f, indent=2)

        meta_url = upload_to_blob(meta_path, "image_faiss")
        logging.info(f"Metadata updated and uploaded: {meta_url}")
    except Exception as e:
        logging.error(f"Failed to handle metadata: {e}")
        raise

def process_pdf_and_create_or_update_vector_db(pdf_path):
    logging.info(f"Starting process_pdf_and_create_or_update_vector_db {pdf_path}")
    texts, images = extract_text_and_images(pdf_path)
    log_mem()
    logging.info("Extracted text and images")

    text_embeds = generate_text_embeddings(texts)
    log_mem()
    logging.info("Generated text embeddings")

    if images:
        # Use the full URL for processing
        img_paths = [img[0] for img in images]

        img_descriptions = [generate_image_description(p) for p in img_paths]
        logging.info("Generated image descriptions")

        img_embeds = generate_image_embeddings_with_context(img_paths, img_descriptions)
        logging.info("Generated image embeddings")

        # Extract only the image names for metadata
        img_names = [os.path.basename(img[0]) for img in images]
    else:
        img_embeds = np.empty((0, 512))
        img_paths, img_descriptions, img_names = [], [], []
        logging.info("No images found in the PDF")

    log_mem()
    create_or_update_vector_db(text_embeds, img_embeds, img_names, img_descriptions)
    logging.info("All vector DBs updated.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python vector_db_generate_update.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf_and_create_or_update_vector_db(pdf_path)