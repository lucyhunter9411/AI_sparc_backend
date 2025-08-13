import os
import json
import torch
import faiss
import fitz  # PyMuPDF
import psutil
import logging
import numpy as np
import shutil
from io import BytesIO
from PIL import Image
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import requests
import tempfile

# Logging setup
logging.basicConfig(level=logging.INFO)

# Local Save Directory
LOCAL_SAVE_DIR = "app/vector_db/vectorstore"
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

# Model Configs
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
SENTENCE_TRANSFORMER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # Keep for other uses

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SENTENCE_TRANSFORMER_MODEL  # Use SentenceTransformer for direct embedding generation
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def log_mem():
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    logging.info(f"Memory usage: {mem:.2f} MB")

def save_to_local(local_path, subdir):
    os.makedirs(os.path.join(LOCAL_SAVE_DIR, subdir), exist_ok=True)
    dest_path = os.path.join(LOCAL_SAVE_DIR, subdir, os.path.basename(local_path))
    shutil.move(local_path, dest_path)
    return dest_path

def extract_text_and_images(pdf_file):
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
            saved_path = save_to_local(local_path, "images")
            images.append((saved_path, page_num, img_idx))
    return texts, images

def generate_image_description(path):
    image = Image.open(path).convert("RGB")
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

# def generate_image_embeddings_with_context(paths, descriptions, target_size=(224, 224)):
#     vectors = []
#     for path, description in zip(paths, descriptions):
#         image = Image.open(path).convert("RGB").resize(target_size)
#         inputs = clip_processor(images=image, text=[description], return_tensors="pt", padding=True)

#         with torch.no_grad():
#             outputs = clip_model(**inputs)
#             image_features = outputs.image_embeds

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
    path = os.path.join(LOCAL_SAVE_DIR, blob_name)
    return path if os.path.exists(path) else None

def load_or_create_index(subdir, index_filename, new_embeddings):
    dim = new_embeddings.shape[1]
    index_path = os.path.join(LOCAL_SAVE_DIR, subdir, index_filename)
    os.makedirs(os.path.join(LOCAL_SAVE_DIR, subdir), exist_ok=True)

    # Normalize the new embeddings first
    faiss.normalize_L2(new_embeddings)
    
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        logging.info(f"Loaded FAISS index from: {index_path}")
        
        # Original logic kept for reference (commented out)
        if index.d != dim:
            logging.warning(f"Dimension mismatch. Creating new index with dim {dim}.")
            index = faiss.IndexFlatIP(dim)  # Changed from IndexFlatL2 to IndexFlatIP
        else:
            # If loading existing index, we need to handle the case where 
            # existing index might be L2 type - recreate as IP type
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
        logging.info(f"No existing index found at {index_path}. Creating new one.")
        index = faiss.IndexFlatIP(dim)  # Changed from IndexFlatL2 to IndexFlatIP

    index.add(new_embeddings)
    return index, index_path

def create_or_update_vector_db(texts, img_embeds, img_paths, img_descriptions):
    # TEXT INDEX - Using Langchain FAISS to create both index.faiss and index.pkl
    try:
        # Create Document objects from texts
        documents = [Document(page_content=text, metadata={"source": f"page_{i}"}) for i, text in enumerate(texts)]
        
        # Create local directory for saving
        local_faiss_dir = os.path.join(LOCAL_SAVE_DIR, "text_faiss")
        os.makedirs(local_faiss_dir, exist_ok=True)
        
        # Check if existing vectorstore exists
        existing_vectorstore = None
        faiss_path = os.path.join(local_faiss_dir, "index.faiss")
        pkl_path = os.path.join(local_faiss_dir, "index.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            try:
                # Load existing vectorstore
                existing_vectorstore = FAISS.load_local(
                    local_faiss_dir,
                    EMBEDDING_MODEL,
                    allow_dangerous_deserialization=True
                )
                logging.info("Loaded existing text FAISS vectorstore from local storage")
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
        logging.info(f"Text FAISS vectorstore saved at: {local_faiss_dir}")
        
    except Exception as e:
        logging.error(f"Failed to create/update text FAISS index: {e}")
        raise

    try:
        image_index, image_path = load_or_create_index("image_faiss", "index.faiss", img_embeds)
        faiss.write_index(image_index, image_path)
        logging.info(f"Image index saved at: {image_path}")
    except Exception as e:
        logging.error(f"Failed to create/update image index: {e}")
        raise

    try:
        meta_path = os.path.join(tempfile.gettempdir(), "image_faiss_metadata.json")
        final_meta_path = os.path.join(LOCAL_SAVE_DIR, "image_faiss", "image_faiss_metadata.json")
        os.makedirs(os.path.dirname(final_meta_path), exist_ok=True)

        existing_metadata = {"image_paths": [], "descriptions": []}
        if os.path.exists(final_meta_path):
            with open(final_meta_path, "r") as f:
                existing_metadata = json.load(f)
                logging.info("Loaded existing metadata")

        updated_metadata = {
            "image_paths": existing_metadata.get("image_paths", []) + img_paths,
            "descriptions": existing_metadata.get("descriptions", []) + img_descriptions
        }

        with open(meta_path, "w") as f:
            json.dump(updated_metadata, f, indent=2)

        save_to_local(meta_path, "image_faiss")
        logging.info(f"Metadata updated and saved at: {final_meta_path}")
    except Exception as e:
        logging.error(f"Failed to save metadata: {e}")
        raise

def process_pdf_and_create_or_update_vector_db(pdf_path):
    logging.info(f"Processing PDF: {pdf_path}")
    texts, images = extract_text_and_images(pdf_path)
    log_mem()

    # No need to generate text embeddings here - Langchain FAISS will handle it
    logging.info("Text will be embedded by Langchain FAISS")

    if images:
        img_paths = [img[0] for img in images]
        img_descriptions = [generate_image_description(p) for p in img_paths]
        img_embeds = generate_image_embeddings_with_context(img_paths, img_descriptions)
        img_names = [os.path.basename(p) for p in img_paths]
    else:
        img_embeds = np.empty((0, 512))
        img_paths, img_descriptions, img_names = [], [], []
        logging.info("No images found in the PDF")

    log_mem()
    create_or_update_vector_db(texts, img_embeds, img_names, img_descriptions)
    logging.info("Finished processing and saving all local vector DB files.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python vector_db_generate_update.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    process_pdf_and_create_or_update_vector_db(pdf_path)
