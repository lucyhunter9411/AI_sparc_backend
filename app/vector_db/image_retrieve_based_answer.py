from dotenv import load_dotenv
from pathlib import Path
# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

import faiss
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import logging
import os
import threading
from app.services.shared_data import set_closest_image_path

# Set up logging
logger = logging.getLogger(__name__)

# Apple Silicon specific configurations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable CPU fallback for MPS
os.environ['TOKENIZERS_PARALLELISM'] = 'false'   # Disable tokenizer parallelism

# Detect Apple Silicon
import platform
if platform.machine() == 'arm64':
    logger.info("Detected Apple Silicon (ARM64) - applying specific configurations")
    # Additional Apple Silicon specific settings
    torch.set_num_threads(1)  # Limit threading to avoid conflicts
else:
    logger.info(f"Detected architecture: {platform.machine()}")

# Thread-local storage for CLIP models
_thread_local = threading.local()

def get_clip_model():
    """Get or create CLIP model for current thread"""
    if not hasattr(_thread_local, 'clip_model'):
        logger.info("Creating CLIP model for current thread")
        _thread_local.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return _thread_local.clip_model

def get_clip_processor():
    """Get or create CLIP processor for current thread"""
    if not hasattr(_thread_local, 'clip_processor'):
        logger.info("Creating CLIP processor for current thread")
        _thread_local.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _thread_local.clip_processor

# Initialize CLIP model and processor in main thread
logger.info("Starting CLIP model initialization...")
try:
    # Initialize in main thread
    get_clip_model()
    get_clip_processor()
    logger.info("CLIP model and processor initialized in main thread")
except Exception as e:
    logger.error(f"Failed to initialize CLIP model in main thread: {e}")
    raise

# Load FAISS index for image description embeddings
# DB_IMAGE_FAISS_PATH = "app/vector_db/vectorstore/image_faiss"  # Update path if necessary
DB_IMAGE_FAISS_PATH = os.getenv("DB_IMAGE_FAISS_PATH", "")
faiss_index_path = f"{DB_IMAGE_FAISS_PATH}.faiss"

logger.info(f"Loading FAISS index from: {faiss_index_path}")
try:
    image_index = faiss.read_index(faiss_index_path)
    logger.info("FAISS index loaded successfully")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    raise

# Load image metadata (paths to images and descriptions)
metadata_path = f"{DB_IMAGE_FAISS_PATH}.json"
logger.info(f"Loading metadata from: {metadata_path}")
try:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    logger.info("Metadata loaded successfully")
except Exception as e:
    logger.error(f"Failed to load metadata: {e}")
    raise

# Normalize image paths to handle cross-platform compatibility
raw_image_paths = metadata["image_paths"]
image_paths = []
for path in raw_image_paths:
    # Convert Windows backslashes to forward slashes and normalize path
    normalized_path = os.path.normpath(path.replace('\\', '/'))
    image_paths.append(normalized_path)
    logger.info(f"Normalized path: {path} -> {normalized_path}")
    
    # Check if the image file exists
    if not os.path.exists(normalized_path):
        logger.warning(f"Image file does not exist: {normalized_path}")

logger.info(f"Loaded {len(image_paths)} image paths")

# Function to retrieve and display multiple images based on user query and generated text
def retrieve_image(user_query, generated_text, robot_id, top_k=1):
    logger.info(f"Starting retrieve_image function with robot_id: {robot_id}")
    logger.info(f"User query: {user_query}")
    logger.info(f"Generated text: {generated_text}")
    logger.info(f"Top k: {top_k}")
    
    try:
        global closest_image_paths
        # Combine user query with generated text (if available)
        combined_text = user_query
        if generated_text:
            combined_text = user_query + " " + generated_text  # Combine user query with the generated text
        
        logger.info(f"Combined text: {combined_text}")

        # Step 1: Generate embedding for the combined text (user query + generated text)
        logger.info("Step 1: Generating text embedding...")
        try:
            # Get thread-local CLIP processor and model
            clip_processor = get_clip_processor()
            clip_model = get_clip_model()
            
            text_inputs = clip_processor(text=[combined_text], return_tensors="pt", padding=True, truncation=True)
            logger.info("Text inputs processed successfully")
        except Exception as e:
            logger.error(f"Failed to process text inputs: {e}")
            raise
        
        try:
            with torch.no_grad():
                query_embedding = clip_model.get_text_features(**text_inputs).numpy().squeeze()
            logger.info("Text embedding generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            raise

        # Step 2: Ensure query_embedding is 2D for FAISS (i.e., shape (1, embedding_dim))
        logger.info("Step 2: Reshaping query embedding...")
        try:
            query_embedding = np.expand_dims(query_embedding, axis=0)  # Make it a 2D array
            logger.info(f"Query embedding shape: {query_embedding.shape}")
        except Exception as e:
            logger.error(f"Failed to reshape query embedding: {e}")
            raise

        # Step 3: Normalize the query embedding
        logger.info("Step 3: Normalizing query embedding...")
        try:
            faiss.normalize_L2(query_embedding)
            logger.info("Query embedding normalized successfully")
        except Exception as e:
            logger.error(f"Failed to normalize query embedding: {e}")
            raise

        # Step 4: Search FAISS for the most similar image description embeddings
        logger.info("Step 4: Searching FAISS index...")
        try:
            D, I = image_index.search(query_embedding, top_k)
            logger.info(f"FAISS search completed. Distances: {D}, Indices: {I}")
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            raise

        # Step 5: Retrieve the closest image paths based on the indices
        logger.info("Step 5: Retrieving closest image paths...")
        print(f"Displaying top {top_k} images for query: '{user_query}'")
        
        closest_image_path = None
        try:
            for idx in I[0]:
                logger.info(f"Processing index: {idx}")
                if idx < len(image_paths):
                    closest_image_path = image_paths[idx]
                    logger.info(f"Setting closest image path: {closest_image_path}")
                    set_closest_image_path(robot_id, closest_image_path)
                    print(f"Displaying image: {closest_image_path}")
                else:
                    logger.error(f"Index {idx} is out of range for image_paths (length: {len(image_paths)})")
        except Exception as e:
            logger.error(f"Failed to process image indices: {e}")
            raise
        
        logger.info(f"retrieve_image function completed successfully. Returning: {closest_image_path}")
        return closest_image_path
        
    except Exception as e:
        logger.error(f"Error in retrieve_image function: {e}", exc_info=True)
        raise

def retrieve_image_safe(user_query, generated_text, robot_id, top_k=1):
    """
    Safe wrapper for retrieve_image that handles PyTorch threading issues on macOS.
    This function loads the CLIP model in the worker thread to avoid threading conflicts.
    """
    logger.info(f"Starting retrieve_image_safe function with robot_id: {robot_id}")
    
    try:
        # Load CLIP model and processor in this thread (worker thread)
        logger.info("Loading CLIP model in worker thread...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Force CPU usage to avoid Apple Silicon MPS issues
        device = torch.device("cpu")
        clip_model = clip_model.to(device)
        logger.info(f"CLIP model loaded successfully in worker thread on device: {device}")
        
        # Combine user query with generated text (if available)
        combined_text = user_query
        if generated_text:
            combined_text = user_query + " " + generated_text  # Combine user query with the generated text
        
        logger.info(f"Combined text: {combined_text}")

        # Step 1: Generate embedding for the combined text (user query + generated text)
        logger.info("Step 1: Generating text embedding...")
        try:
            text_inputs = clip_processor(text=[combined_text], return_tensors="pt", padding=True, truncation=True)
            # Move inputs to CPU
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            logger.info("Text inputs processed successfully")
        except Exception as e:
            logger.error(f"Failed to process text inputs: {e}")
            raise
        
        try:
            with torch.no_grad():
                query_embedding = clip_model.get_text_features(**text_inputs).cpu().numpy().squeeze()
            logger.info("Text embedding generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            raise

        # Step 2: Ensure query_embedding is 2D for FAISS (i.e., shape (1, embedding_dim))
        logger.info("Step 2: Reshaping query embedding...")
        try:
            query_embedding = np.expand_dims(query_embedding, axis=0)  # Make it a 2D array
            logger.info(f"Query embedding shape: {query_embedding.shape}")
        except Exception as e:
            logger.error(f"Failed to reshape query embedding: {e}")
            raise

        # Step 3: Normalize the query embedding
        logger.info("Step 3: Normalizing query embedding...")
        try:
            faiss.normalize_L2(query_embedding)
            logger.info("Query embedding normalized successfully")
        except Exception as e:
            logger.error(f"Failed to normalize query embedding: {e}")
            raise

        # Step 4: Search FAISS for the most similar image description embeddings
        logger.info("Step 4: Searching FAISS index...")
        try:
            D, I = image_index.search(query_embedding, top_k)
            logger.info(f"FAISS search completed. Distances: {D}, Indices: {I}")
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            raise

        # Step 5: Retrieve the closest image paths based on the indices
        logger.info("Step 5: Retrieving closest image paths...")
        print(f"Displaying top {top_k} images for query: '{user_query}'")
        
        closest_image_path = None
        try:
            for idx in I[0]:
                logger.info(f"Processing index: {idx}")
                if idx < len(image_paths):
                    closest_image_path = image_paths[idx]
                    logger.info(f"Setting closest image path: {closest_image_path}")
                    set_closest_image_path(robot_id, closest_image_path)
                    print(f"Displaying image: {closest_image_path}")
                else:
                    logger.error(f"Index {idx} is out of range for image_paths (length: {len(image_paths)})")
        except Exception as e:
            logger.error(f"Failed to process image indices: {e}")
            raise
        
        logger.info(f"retrieve_image_safe function completed successfully. Returning: {closest_image_path}")
        return closest_image_path
        
    except Exception as e:
        logger.error(f"Error in retrieve_image_safe function: {e}", exc_info=True)
        return None


