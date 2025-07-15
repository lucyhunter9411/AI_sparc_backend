from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

import faiss
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import logging
import os
import threading
import platform
import requests
import tempfile
from app.services.shared_data import set_closest_image_path
from googletrans import Translator

# Set up logging
logger = logging.getLogger(__name__)

# Apple Silicon specific configurations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if platform.machine() == 'arm64':
    logger.info("Detected Apple Silicon (ARM64) - applying specific configurations")
    torch.set_num_threads(1)
else:
    logger.info(f"Detected architecture: {platform.machine()}")

_thread_local = threading.local()

def get_clip_model():
    if not hasattr(_thread_local, 'clip_model'):
        logger.info("Creating CLIP model for current thread")
        _thread_local.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return _thread_local.clip_model

def get_clip_processor():
    if not hasattr(_thread_local, 'clip_processor'):
        logger.info("Creating CLIP processor for current thread")
        _thread_local.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _thread_local.clip_processor

logger.info("Starting CLIP model initialization...")
try:
    get_clip_model()
    get_clip_processor()
    logger.info("CLIP model and processor initialized in main thread")
except Exception as e:
    logger.error(f"Failed to initialize CLIP model in main thread: {e}")
    raise

async def retrieve_image_safe(user_query, generated_text, robot_id, top_k=1):
    logger.info(f"Starting retrieve_image_safe function with robot_id: {robot_id}")
    try:
        # Initialize FAISS index and metadata
        DB_IMAGE_FAISS_PATH = os.getenv("DB_IMAGE_FAISS_PATH", "https://classroomdata.blob.core.windows.net/pdf-images/image_faiss")
        UPLOAD_FOLDER_FAISS = os.getenv("UPLOAD_FOLDER_FAISS")
        local_faiss_path = os.path.join(tempfile.gettempdir(), "index.faiss")
        local_json_path = os.path.join(tempfile.gettempdir(), "image_faiss_metadata.json")

        faiss_url = f"{DB_IMAGE_FAISS_PATH}/index.faiss"
        logger.info(f"Downloading FAISS index from: {faiss_url}")
        response = requests.get(faiss_url)
        response.raise_for_status()
        with open(local_faiss_path, "wb") as f:
            f.write(response.content)
        image_index = faiss.read_index(local_faiss_path)
        logger.info("FAISS index downloaded and loaded successfully")

        metadata_url = f"{DB_IMAGE_FAISS_PATH}/image_faiss_metadata.json"
        logger.info(f"Downloading metadata from: {metadata_url}")
        response = requests.get(metadata_url)
        response.raise_for_status()
        with open(local_json_path, "wb") as f:
            f.write(response.content)
        with open(local_json_path, "r") as f:
            metadata = json.load(f)
        logger.info("Metadata downloaded and loaded successfully")

        raw_image_paths = metadata["image_paths"]
        image_paths = []
        for path in raw_image_paths:
            # normalized_path = os.path.normpath(path.replace('\\', '/'))
            # image_paths.append(normalized_path)
            # logger.info(f"Normalized path: {path} -> {normalized_path}")
            # if not os.path.exists(normalized_path):
            #     logger.warning(f"Image file does not exist: {normalized_path}")
            image_paths.append(f"{UPLOAD_FOLDER_FAISS}/images/{path}")

        # Process the query
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(torch.device("cpu"))
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        combined_text = user_query + " " + generated_text if generated_text else user_query
        logger.info(f"Combined text: {combined_text}")
        
        # Translate the combined text to English
        translator = Translator()
        translation = await translator.translate(combined_text, dest='en')
        translated_text = translation.text
        logger.info(f"Translated text: {translated_text}")

        text_inputs = clip_processor(text=[combined_text], return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(torch.device("cpu")) for k, v in text_inputs.items()}
        with torch.no_grad():
            query_embedding = clip_model.get_text_features(**text_inputs).cpu().numpy().squeeze()

        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)
        D, I = image_index.search(query_embedding, top_k)
        logger.info(f"FAISS search completed. Distances: {D}, Indices: {I}")

        closest_image_path = None
        for idx in I[0]:
            logger.info(f"Processing index: {idx}")
            if idx < len(image_paths):
                closest_image_path = image_paths[idx]
                logger.info(f"Setting closest image path: {closest_image_path}")
                set_closest_image_path(robot_id, closest_image_path)
                print(f"Displaying image: {closest_image_path}")
            else:
                logger.error(f"Index {idx} is out of range for image_paths (length: {len(image_paths)})")

        logger.info(f"retrieve_image_safe function completed successfully. Returning: {closest_image_path}")
        return closest_image_path

    except Exception as e:
        logger.error(f"Error in retrieve_image_safe function: {e}", exc_info=True)
        return None