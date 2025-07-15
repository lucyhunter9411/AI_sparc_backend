import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
from io import BytesIO
import json
import pickle
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Paths
# IMAGE_DB_PATH = "app/vector_db/vectorstore/image_faiss"
# DB_TEXT_FAISS_PATH = "app/vector_db/vectorstore/text_faiss"
# IMAGE_DIR = "app/vector_db/images"
DB_TEXT_FAISS_PATH = os.getenv("DB_TEXT_FAISS_PATH", "")
IMAGE_DB_PATH = os.getenv("DB_IMAGE_FAISS_PATH", "")
IMAGE_DIR = os.getenv("IMAGE_DIR", "")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DB_TEXT_FAISS_PATH, exist_ok=True)

# Create the 'vectorstore' directory if it doesn't exist
if not os.path.exists('app/vector_db/vectorstore'):
    os.makedirs('app/vector_db/vectorstore')

# Extract text and images from the PDF
def extract_text_and_images(pdf_file):
    # Extract text and images using PyMuPDF (fitz)
    logging.info(f"Extracting text and images from {pdf_file}")
    doc = fitz.open(pdf_file)
    text_data = []
    image_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extract text
        text_data.append(page.get_text())
        
        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            # Get the current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = os.path.join(IMAGE_DIR, f"image_{page_num}_{img_index}_{timestamp}.png")
            image.save(image_path)  # Save the image
            image_data.append((image_path, page_num, img_index))
    
    return text_data, image_data

def generate_image_description(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Process the image and generate a caption
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    description = blip_processor.decode(out[0], skip_special_tokens=True)
    
    return description

# Generate text embeddings
def generate_text_embeddings(text_data):
    text_embeddings_list = []
    for text in text_data:
        embeddings = text_embeddings.encode(text)
        text_embeddings_list.append(embeddings)
    return np.array(text_embeddings_list)

# Generate image embeddings
def generate_image_embeddings(image_paths):
    image_embeddings_list = []
    for img_path in image_paths:
        image = Image.open(img_path)
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        image_embeddings_list.append(image_features.numpy().squeeze())
    return np.array(image_embeddings_list)

# Load or create the FAISS index and add both text and image embeddings
def load_or_create_index(existing_index_path, embeddings, is_image=False):
    # Check if the FAISS index exists
    if os.path.exists(existing_index_path):
        # Load the existing FAISS index
        index = faiss.read_index(existing_index_path)
        logging.info("Existing FAISS index loaded.")
    else:
        # Create a new FAISS index if it doesn't exist
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        logging.info("New FAISS index created.")
    # Add the new embeddings to the index
    index.add(embeddings)
    return index

# Create or update the FAISS index
def create_or_update_vector_db(text_embeddings, image_embeddings, image_paths, image_descriptions):
    # Load or create the text FAISS index
    text_index = load_or_create_index(os.path.join(DB_TEXT_FAISS_PATH, "index.faiss"), text_embeddings)
    
    # Load or create the image FAISS index
    image_index = load_or_create_index(f"{IMAGE_DB_PATH}.faiss", image_embeddings)
    
    # Save the updated FAISS index
    faiss.write_index(text_index, os.path.join(DB_TEXT_FAISS_PATH, "index.faiss"))
    faiss.write_index(image_index, f"{IMAGE_DB_PATH}.faiss")
    
    # Load the existing metadata, if any, and update
    metadata_path = f"{IMAGE_DB_PATH}.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "image_paths": [],
            "descriptions": []
        }
    
    # Add new metadata
    metadata["image_paths"].extend(image_paths)
    metadata["descriptions"].extend(image_descriptions)
    
    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    logging.info("Vector database and metadata updated successfully.")

# Example usage within your existing workflow
def process_pdf_and_create_or_update_vector_db(pdf_file):
    logging.info("Starting process_pdf_and_create_or_update_vector_db")
    # Step 1: Extract text and images from the PDF
    text_data, image_data = extract_text_and_images(pdf_file)
    logging.info("Extracted text and images")
    
    # Step 2: Generate embeddings for text
    text_embeddings_array = generate_text_embeddings(text_data)
    logging.info("Generated text embeddings")
    
    # Step 3: Check if there are any images to process
    if image_data:
        # Generate embeddings for images
        image_paths = [img[0] for img in image_data]  # Extract image paths from image_data
        image_embeddings_array = generate_image_embeddings(image_paths)
        logging.info("Generated image embeddings")
        
        # Automatically generate descriptions for each image
        image_descriptions = [generate_image_description(img_path) for img_path in image_paths]
        logging.info("Generated image descriptions")
    else:
        # If no images, initialize empty arrays
        image_embeddings_array = np.empty((0, 512))  # Assuming 512 is the embedding dimension
        image_paths = []
        image_descriptions = []
        logging.info("No images found in the PDF")
    
    # Step 5: Create or update the FAISS index for text and image embeddings, and save metadata
    create_or_update_vector_db(text_embeddings_array, image_embeddings_array, image_paths, image_descriptions)
    logging.info("Updated vector database and metadata successfully")

# Run the process with the new PDF file
# process_pdf_and_create_or_update_vector_db(pdf_file)
