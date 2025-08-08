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
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from azure.storage.blob import BlobServiceClient, ContentSettings
import requests
import tempfile

# Logging setup
logging.basicConfig(level=logging.INFO)

# Azure Blob configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
SENTENCE_TRANSFORMER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # Keep for other uses
BLOB_CONTAINER_NAME = "pdf-images"

# Azure Blob client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

def upload_to_blob(local_path, blob_subdir):
    blob_name = f"{blob_subdir}/{os.path.basename(local_path)}"
    content_settings = ContentSettings(content_type="image/jpeg")  # ← Changed from "application/octet-stream"
    with open(local_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True, content_settings=content_settings)
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob_name}"
    return blob_url

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SENTENCE_TRANSFORMER_MODEL  # Use SentenceTransformer for direct embedding generation
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
            
            filename = f"image_{page_num}_{img_idx}_{pdf_file_name}.jpg"
            local_path = os.path.join(tempfile.gettempdir(), filename)
            image.save(local_path, format="JPEG", quality=50, optimize=True)
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

# Note: generate_text_embeddings function removed - Langchain FAISS handles text embeddings internally

# def generate_image_embeddings_with_context(paths, descriptions, target_size=(224, 224)):
#     vectors = []
#     for path, description in zip(paths, descriptions):
#         response = requests.get(path)
#         image = Image.open(BytesIO(response.content)).convert("RGB").resize(target_size)
#
#         # 👇 Use both image + description
#         inputs = clip_processor(images=image, text=[description], return_tensors="pt", padding=True)
#
#         with torch.no_grad():
#             outputs = clip_model(**inputs)  # ✅ call the full model
#             image_features = outputs.image_embeds  # ✅ get the guided image embedding
#
#         vectors.append(image_features.cpu().numpy().squeeze())
#         del inputs, outputs
#         torch.cuda.empty_cache()
#     return np.array(vectors)

def generate_image_embeddings_with_context(paths, descriptions, target_size=(224, 224)):
    """
    Generate text-only embeddings for image descriptions.
    This ensures compatibility with text queries during retrieval.
    """
    vectors = []
    for path, description in zip(paths, descriptions):
        # Use only the text description, not the image
        # This creates embeddings in the same space as query text embeddings
        inputs = clip_processor(text=[description], return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            # Use get_text_features instead of the full model
            text_features = clip_model.get_text_features(**inputs)
        
        vectors.append(text_features.cpu().numpy().squeeze())
        del inputs, text_features
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

    # Normalize the new embeddings first
    faiss.normalize_L2(new_embeddings)

    downloaded = download_blob_to_temp(blob_name)

    if downloaded and os.path.exists(temp_path):
        index = faiss.read_index(temp_path)
        logging.info(f"Loaded FAISS index from blob: {blob_name}")

        if index.d != dim:
            logging.warning(f"Index dimension mismatch: found {index.d}, expected {dim}. Creating new index.")
            index = faiss.IndexFlatIP(dim)  # Changed from IndexFlatL2
        else:
            # Handle existing L2 indexes - convert to IP
            if hasattr(index, 'metric_type') and index.metric_type != faiss.METRIC_INNER_PRODUCT:
                logging.info("Converting existing L2 index to Inner Product index")
                # Extract all existing vectors
                existing_vectors = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])
                # Normalize existing vectors
                faiss.normalize_L2(existing_vectors)
                # Create new IP index and add normalized vectors
                index = faiss.IndexFlatIP(dim)
                index.add(existing_vectors)
    else:
        logging.info("No existing index found. Creating new one.")
        index = faiss.IndexFlatIP(dim)  # Changed from IndexFlatL2

    index.add(new_embeddings)
    return index, temp_path

def create_or_update_vector_db(texts, img_embeds, img_paths, img_descriptions):
    # TEXT INDEX - Using Langchain FAISS to create both index.faiss and index.pkl
    try:
        # Create Document objects from texts
        documents = [Document(page_content=text, metadata={"source": f"page_{i}"}) for i, text in enumerate(texts)]
        
        # Create local directory for saving
        temp_dir = tempfile.gettempdir()
        local_faiss_dir = os.path.join(temp_dir, "text_faiss_temp")
        os.makedirs(local_faiss_dir, exist_ok=True)
        
        # Check if existing index exists and download it
        existing_vectorstore = None
        try:
            # Try to download existing files
            faiss_blob_name = "text_faiss/index.faiss"
            pkl_blob_name = "text_faiss/index.pkl"
            
            faiss_downloaded = download_blob_to_temp(faiss_blob_name)
            pkl_downloaded = download_blob_to_temp(pkl_blob_name)
            
            # Copy downloaded files to our working directory
            if faiss_downloaded and pkl_downloaded:
                import shutil
                faiss_temp_path = os.path.join(tempfile.gettempdir(), "index.faiss")
                pkl_temp_path = os.path.join(tempfile.gettempdir(), "index.pkl")
                
                if os.path.exists(faiss_temp_path) and os.path.exists(pkl_temp_path):
                    shutil.copy(faiss_temp_path, os.path.join(local_faiss_dir, "index.faiss"))
                    shutil.copy(pkl_temp_path, os.path.join(local_faiss_dir, "index.pkl"))
                    
                    # Load existing vectorstore
                    existing_vectorstore = FAISS.load_local(
                        local_faiss_dir,
                        EMBEDDING_MODEL,
                        allow_dangerous_deserialization=True
                    )
                    logging.info("Loaded existing text FAISS vectorstore from blob")
        except Exception as e:
            logging.warning(f"Could not load existing text vectorstore: {e}")
        
        # Create or update vectorstore
        if existing_vectorstore:
            # Add new documents to existing vectorstore
            existing_vectorstore.add_documents(documents)
            vectorstore = existing_vectorstore
            logging.info("Added new documents to existing text vectorstore")
        else:
            # Create new vectorstore from documents
            vectorstore = FAISS.from_documents(documents, EMBEDDING_MODEL)
            logging.info("Created new text vectorstore from documents")
        
        # Save vectorstore locally (creates both index.faiss and index.pkl)
        vectorstore.save_local(local_faiss_dir)
        
        # Upload both files to blob storage
        faiss_path = os.path.join(local_faiss_dir, "index.faiss")
        pkl_path = os.path.join(local_faiss_dir, "index.pkl")
        
        faiss_url = upload_to_blob(faiss_path, "text_faiss")
        pkl_url = upload_to_blob(pkl_path, "text_faiss")
        
        logging.info(f"Text FAISS index uploaded: {faiss_url}")
        logging.info(f"Text FAISS pkl uploaded: {pkl_url}")
        
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

    # No need to generate text embeddings here - Langchain FAISS will handle it
    logging.info("Text will be embedded by Langchain FAISS")

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
    create_or_update_vector_db(texts, img_embeds, img_names, img_descriptions)
    logging.info("All vector DBs updated.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python vector_db_generate_update.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf_and_create_or_update_vector_db(pdf_path)
