# import os
# import json
# import torch
# import faiss
# import fitz  # PyMuPDF
# import psutil
# import logging
# import numpy as np
# from io import BytesIO
# from PIL import Image
# from datetime import datetime
# from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
# from sentence_transformers import SentenceTransformer

# # Logging setup
# logging.basicConfig(level=logging.INFO)

# # Paths
# DB_TEXT_FAISS_PATH = os.getenv("DB_TEXT_FAISS_PATH", "")
# IMAGE_DB_PATH = os.getenv("DB_IMAGE_FAISS_PATH", "")
# IMAGE_DIR = os.getenv("IMAGE_DIR", "")

# # Ensure dirs exist
# os.makedirs(IMAGE_DIR, exist_ok=True)
# os.makedirs(DB_TEXT_FAISS_PATH, exist_ok=True)
# os.makedirs(os.path.dirname(IMAGE_DB_PATH), exist_ok=True)

# # Load models
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# text_model = SentenceTransformer('all-MiniLM-L6-v2')
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# def log_mem():
#     mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
#     logging.info(f"Memory usage: {mem:.2f} MB")

# # Create the 'vectorstore' directory if it doesn't exist
# if not os.path.exists('app/vector_db/vectorstore'):
#     os.makedirs('app/vector_db/vectorstore')

# # Extract text and images from the PDF
# def extract_text_and_images(pdf_file):
#     doc = fitz.open(pdf_file)
#     texts, images = [], []
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         texts.append(page.get_text())
#         for img_idx, img in enumerate(page.get_images(full=True)):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             img_bytes = base_image["image"]
#             image = Image.open(BytesIO(img_bytes)).convert("RGB")
#             filename = f"image_{page_num}_{img_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
#             path = os.path.join(IMAGE_DIR, filename)
#             image.save(path)
#             images.append((path, page_num, img_idx))
#     return texts, images

# def generate_image_description(path):
#     image = Image.open(path).convert("RGB")
#     inputs = blip_processor(image, return_tensors="pt")
#     out = blip_model.generate(**inputs)
#     desc = blip_processor.decode(out[0], skip_special_tokens=True)
#     del image, inputs, out
#     torch.cuda.empty_cache()
#     return desc

# # Generate text embeddings
# def generate_text_embeddings(texts, batch_size=16):
#     embeddings = []
#     for i in range(0, len(texts), batch_size):
#         chunk = texts[i:i+batch_size]
#         emb = text_model.encode(chunk, show_progress_bar=False)
#         embeddings.extend(emb)
#     return np.array(embeddings)

# # Generate image embeddings
# def generate_image_embeddings(paths, target_size=(224, 224)):
#     vectors = []
#     for path in paths:
#         image = Image.open(path).convert("RGB").resize(target_size)
#         inputs = clip_processor(images=image, return_tensors="pt", padding=True)
#         with torch.no_grad():
#             features = clip_model.get_image_features(**inputs)
#         vectors.append(features.cpu().numpy().squeeze())
#         del inputs, features
#         torch.cuda.empty_cache()
#     return np.array(vectors)

# # Load or create the FAISS index and add both text and image embeddings
# def load_or_create_index(path, embeddings):
#     # Check if the FAISS index exists
#     if os.path.exists(path):
#         # Load the existing FAISS index
#         index = faiss.read_index(path)
#         logging.info("Loaded existing FAISS index")
#     else:
#         # Create a new FAISS index if it doesn't exist
#         index = faiss.IndexFlatL2(embeddings.shape[1])
#         logging.info("Created new FAISS index")
#     # Add the new embeddings to the index
#     index.add(embeddings)
#     return index

# # Create or update the FAISS index
# def create_or_update_vector_db(text_embeds, img_embeds, img_paths, img_descriptions):
#     # Load or create the text FAISS index
#     text_index = load_or_create_index(os.path.join(DB_TEXT_FAISS_PATH, "index.faiss"), text_embeds)
    
#     # Load or create the image FAISS index
#     image_index = load_or_create_index(f"{IMAGE_DB_PATH}.faiss", img_embeds)

#     # Save the updated FAISS index
#     faiss.write_index(text_index, os.path.join(DB_TEXT_FAISS_PATH, "index.faiss"))
#     faiss.write_index(image_index, f"{IMAGE_DB_PATH}.faiss")

#     # Load the existing metadata, if any, and update
#     meta_path = f"{IMAGE_DB_PATH}.json"
#     metadata = {"image_paths": [], "descriptions": []}
#     if os.path.exists(meta_path):
#         with open(meta_path, "r") as f:
#             metadata = json.load(f)

#     # Add new metadata
#     metadata["image_paths"].extend(img_paths)
#     metadata["descriptions"].extend(img_descriptions)

#     # Save updated metadata
#     with open(meta_path, "w") as f:
#         json.dump(metadata, f, indent=2)
        
#     logging.info("Vector database and metadata updated successfully.")

# # Example usage within your existing workflow
# def process_pdf_and_create_or_update_vector_db(pdf_path):
#     logging.info(f"Starting process_pdf_and_create_or_update_vector_db {pdf_path}")
    
#     # Step 1: Extract text and images from the PDF
#     texts, images = extract_text_and_images(pdf_path)
#     log_mem()
#     logging.info("Extracted text and images")

#     # Step 2: Generate embeddings for text
#     text_embeds = generate_text_embeddings(texts)
#     log_mem()
#     logging.info("Generated text embeddings")

#     # Step 3: Check if there are any images to process
#     if images:
#         # Generate embeddings for images
#         img_paths = [img[0] for img in images]
#         img_embeds = generate_image_embeddings(img_paths)
#         logging.info("Generated image embeddings")
        
#         # Automatically generate descriptions for each image
#         img_descriptions = [generate_image_description(p) for p in img_paths]
#         logging.info("Generated image descriptions")
#     else:
#         # If no images, initialize empty arrays
#         img_embeds = np.empty((0, 512))
#         img_paths, img_descriptions = [], []
#         logging.info("No images found in the PDF")

#     # Step 5: Create or update the FAISS index for text and image embeddings, and save metadata
#     log_mem()
#     create_or_update_vector_db(text_embeds, img_embeds, img_paths, img_descriptions)
#     logging.info("All vector DBs updated.")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python vector_db_generate_update.py <pdf_path>")
#         sys.exit(1)

#     pdf_path = sys.argv[1]
#     process_pdf_and_create_or_update_vector_db(pdf_path)

# # Example usage
# # process_pdf_and_create_or_update_vector_db("sample.pdf")


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
from azure.storage.blob import BlobServiceClient

# Logging setup
logging.basicConfig(level=logging.INFO)

# Paths
DB_TEXT_FAISS_PATH = os.getenv("DB_TEXT_FAISS_PATH", "")
IMAGE_DB_PATH = os.getenv("DB_IMAGE_FAISS_PATH", "")
IMAGE_DIR = os.getenv("IMAGE_DIR", "")
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "pdf-images"

# Ensure dirs exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DB_TEXT_FAISS_PATH, exist_ok=True)
os.makedirs(os.path.dirname(IMAGE_DB_PATH), exist_ok=True)

# Azure Blob client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

def upload_image_to_blob(local_path):
    blob_name = os.path.basename(local_path)
    with open(local_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)
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

# Create the 'vectorstore' directory if it doesn't exist
if not os.path.exists('app/vector_db/vectorstore'):
    os.makedirs('app/vector_db/vectorstore')

# Extract text and images from the PDF
def extract_text_and_images(pdf_file):
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
            filename = f"image_{page_num}_{img_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            path = os.path.join(IMAGE_DIR, filename)
            image.save(path)
            public_url = upload_image_to_blob(path)
            images.append((public_url, page_num, img_idx))
    return texts, images

def generate_image_description(path):
    image = Image.open(path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    desc = blip_processor.decode(out[0], skip_special_tokens=True)
    del image, inputs, out
    torch.cuda.empty_cache()
    return desc

# Generate text embeddings
def generate_text_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        emb = text_model.encode(chunk, show_progress_bar=False)
        embeddings.extend(emb)
    return np.array(embeddings)

# Generate image embeddings
def generate_image_embeddings(paths, target_size=(224, 224)):
    vectors = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize(target_size)
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        vectors.append(features.cpu().numpy().squeeze())
        del inputs, features
        torch.cuda.empty_cache()
    return np.array(vectors)

# Load or create the FAISS index and add both text and image embeddings
def load_or_create_index(path, embeddings):
    if os.path.exists(path):
        index = faiss.read_index(path)
        logging.info("Loaded existing FAISS index")
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        logging.info("Created new FAISS index")
    index.add(embeddings)
    return index

# Create or update the FAISS index
def create_or_update_vector_db(text_embeds, img_embeds, img_paths, img_descriptions):
    text_index = load_or_create_index(os.path.join(DB_TEXT_FAISS_PATH, "index.faiss"), text_embeds)
    image_index = load_or_create_index(f"{IMAGE_DB_PATH}.faiss", img_embeds)
    faiss.write_index(text_index, os.path.join(DB_TEXT_FAISS_PATH, "index.faiss"))
    faiss.write_index(image_index, f"{IMAGE_DB_PATH}.faiss")

    meta_path = f"{IMAGE_DB_PATH}.json"
    metadata = {"image_paths": [], "descriptions": []}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    metadata["image_paths"].extend(img_paths)
    metadata["descriptions"].extend(img_descriptions)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Vector database and metadata updated successfully.")

def process_pdf_and_create_or_update_vector_db(pdf_path):
    logging.info(f"Starting process_pdf_and_create_or_update_vector_db {pdf_path}")
    texts, images = extract_text_and_images(pdf_path)
    log_mem()
    logging.info("Extracted text and images")

    text_embeds = generate_text_embeddings(texts)
    log_mem()
    logging.info("Generated text embeddings")

    if images:
        img_paths = [img[0] for img in images]
        img_embeds = generate_image_embeddings(img_paths)
        logging.info("Generated image embeddings")

        img_descriptions = [generate_image_description(p) for p in img_paths]
        logging.info("Generated image descriptions")
    else:
        img_embeds = np.empty((0, 512))
        img_paths, img_descriptions = [], []
        logging.info("No images found in the PDF")

    log_mem()
    create_or_update_vector_db(text_embeds, img_embeds, img_paths, img_descriptions)
    logging.info("All vector DBs updated.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python vector_db_generate_update.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf_and_create_or_update_vector_db(pdf_path)

# Example usage
# process_pdf_and_create_or_update_vector_db("sample.pdf")
