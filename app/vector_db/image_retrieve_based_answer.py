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
import time
from PIL import Image
from app.services.shared_data import set_closest_image_path
from googletrans import Translator

# Set up logging
logger = logging.getLogger(__name__)

# Apple Silicon specific configurations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
LOCAL_MODE = os.getenv("LOCAL_MODE", "0").lower() in ("1", "true", "yes")

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

def analyze_image_retrieval_performance(timings: dict, robot_id: str, local_mode: bool):
    """Analyze image retrieval performance and provide insights."""
    total = timings["total"]
    
    if local_mode:
        components = {
            "Loading": timings.get("loading", 0),
            "CLIP Init": timings.get("clip_init", 0),
            "Translation": timings.get("translation", 0),
            "Embedding": timings.get("embedding", 0),
            "Search": timings.get("search", 0),
            "Result Processing": timings.get("result_processing", 0)
        }
    else:
        components = {
            "Download": timings.get("download", 0),
            "CLIP Init": timings.get("clip_init", 0),
            "Translation": timings.get("translation", 0),
            "Embedding": timings.get("embedding", 0),
            "Search": timings.get("search", 0),
            "Result Processing": timings.get("result_processing", 0)
        }
    
    # Find the slowest component
    slowest = max(components.items(), key=lambda x: x[1])
    
    # Calculate percentages 
    percentages = {name: (duration/total)*100 for name, duration in components.items() if duration > 0}
    
    logger.info(f"[{robot_id}] 📊 Image Retrieval Analysis - Slowest: {slowest[0]} ({percentages.get(slowest[0], 0):.1f}%, {slowest[1]:.3f}s)")
    
    # Provide optimization suggestions
    if slowest[0] == "CLIP Init" and slowest[1] > 1.0:
        logger.info(f"[{robot_id}] 💡 Suggestion: Consider pre-loading CLIP model to reduce initialization time")
    elif slowest[0] == "Download" and slowest[1] > 2.0:
        logger.info(f"[{robot_id}] 💡 Suggestion: Consider using local FAISS index for better performance")
    elif slowest[0] == "Translation" and slowest[1] > 0.5:
        logger.info(f"[{robot_id}] 💡 Suggestion: Consider caching translations or using offline translation")

async def retrieve_image_safe(user_query, generated_text, robot_id, top_k=1):
    retrieval_start = time.perf_counter()
    logger.info(f"[{robot_id}] 🖼️  Starting image retrieval for query: {user_query[:30]}...")
    
    try:
        # Initialize timing variables
        loading_duration = 0
        download_duration = 0
        
        # Initialize FAISS index and metadata
        BLOB_STORAGE_IMAGE_FAISS_DIR = os.getenv("BLOB_STORAGE_IMAGE_FAISS_DIR")
        BLOB_STORAGE_FAISS_DIR = os.getenv("BLOB_STORAGE_FAISS_DIR")

        if LOCAL_MODE:
            # Use local files
            loading_start = time.perf_counter()
            local_faiss_path = "app/vector_db/vectorstore/image_faiss/index.faiss"
            local_json_path = "app/vector_db/vectorstore/image_faiss/image_faiss_metadata.json"
            
            faiss_load_start = time.perf_counter()
            image_index = faiss.read_index(local_faiss_path)
            faiss_load_duration = time.perf_counter() - faiss_load_start
            
            metadata_load_start = time.perf_counter()
            with open(local_json_path, "r") as f:
                metadata = json.load(f)
            metadata_load_duration = time.perf_counter() - metadata_load_start
            
            loading_duration = time.perf_counter() - loading_start
            logger.info(f"[{robot_id}] 📁 Local files loaded in %.3fs (FAISS: %.3fs, Metadata: %.3fs)", 
                       loading_duration, faiss_load_duration, metadata_load_duration)

            raw_image_paths = metadata["image_paths"]
            image_paths = []
            for path in raw_image_paths:
                # Normalize slashes
                normalized_path = os.path.normpath(path.replace('\\', '/'))
                # Remove the leading 'app/vector_db/images/' if present
                if normalized_path.startswith("app" + os.sep + "vector_db" + os.sep + "images" + os.sep):
                    url_path = "images" + os.sep + normalized_path.split(os.sep, 4)[-1]
                else:
                    # fallback: just use the filename
                    url_path = "images" + os.sep + os.path.basename(normalized_path)
                image_paths.append(f"http://localhost:8000/{url_path.replace(os.sep, '/')}")
                if not os.path.exists(normalized_path):
                    logger.warning(f"Image file does not exist: {normalized_path}")
        else:
            # Download from Azure
            download_start = time.perf_counter()
            local_faiss_path = os.path.join(tempfile.gettempdir(), "index.faiss")
            local_json_path = os.path.join(tempfile.gettempdir(), "image_faiss_metadata.json")

            # Download FAISS index
            faiss_download_start = time.perf_counter()
            faiss_url = f"{BLOB_STORAGE_IMAGE_FAISS_DIR}/index.faiss"
            response = requests.get(faiss_url)
            response.raise_for_status()
            with open(local_faiss_path, "wb") as f:
                f.write(response.content)
            image_index = faiss.read_index(local_faiss_path)
            faiss_download_duration = time.perf_counter() - faiss_download_start

            # Download metadata
            metadata_download_start = time.perf_counter()
            metadata_url = f"{BLOB_STORAGE_IMAGE_FAISS_DIR}/image_faiss_metadata.json"
            response = requests.get(metadata_url)
            response.raise_for_status()
            with open(local_json_path, "wb") as f:
                f.write(response.content)
            with open(local_json_path, "r") as f:
                metadata = json.load(f)
            metadata_download_duration = time.perf_counter() - metadata_download_start
            
            download_duration = time.perf_counter() - download_start
            logger.info(f"[{robot_id}] ☁️  Remote files downloaded in %.3fs (FAISS: %.3fs, Metadata: %.3fs)", 
                       download_duration, faiss_download_duration, metadata_download_duration)
            
            raw_image_paths = metadata["image_paths"]
            image_paths = []
            for path in raw_image_paths:
                # normalized_path = os.path.normpath(path.replace('\\', '/'))
                # image_paths.append(normalized_path)
                # logger.info(f"Normalized path: {path} -> {normalized_path}")
                # if not os.path.exists(normalized_path):
                #     logger.warning(f"Image file does not exist: {normalized_path}")
                image_paths.append(f"{BLOB_STORAGE_FAISS_DIR}/images/{path}")


        # Process the query
        processing_start = time.perf_counter()
        
        # Initialize CLIP model and processor
        clip_init_start = time.perf_counter()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(torch.device("cpu"))
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_init_duration = time.perf_counter() - clip_init_start

        # Use only user query for image search to avoid LLM response contamination
        search_text = user_query
        
        # Translate the search text to English if needed
        translation_start = time.perf_counter()
        translated_text = search_text  # Default to original text
        try:
            translator = Translator()
            translation_result = translator.translate(search_text, dest='en')
            if hasattr(translation_result, 'text') and translation_result.text:
                translated_text = translation_result.text
            else:
                logger.warning(f"[{robot_id}] Translation result has no text attribute, using original text")
        except Exception as translation_error:
            logger.warning(f"[{robot_id}] Translation failed: {translation_error}, using original text")
        
        translation_duration = time.perf_counter() - translation_start
        logger.info(f"[{robot_id}] 🌐 Translation completed in %.3fs: {translated_text}", translation_duration)
        logger.info(f"[{robot_id}] 🔍 Using query for image search: '{translated_text}'")

        # Generate embeddings using the SAME multimodal approach as during indexing
        embedding_start = time.perf_counter()
        
        # CRITICAL: Must match the indexing method exactly to get meaningful FAISS scores
        # Index was created with: clip_processor(images=image, text=[description])
        # So query must also use: clip_processor(images=dummy_image, text=[query])
        
        # Create a neutral dummy image (same as indexing approach)
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))  # Neutral gray
        
        # Use the EXACT same method as indexing: image + text -> image_embeds
        inputs = clip_processor(images=dummy_image, text=[translated_text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            # Use image_embeds to match the indexing approach exactly
            query_embedding = outputs.image_embeds.cpu().numpy().squeeze()
        
        embedding_duration = time.perf_counter() - embedding_start
        logger.info(f"[{robot_id}] 🔧 Using multimodal embedding (dummy image + text) to match index creation method")

        # Perform FAISS search
        search_start = time.perf_counter()
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)
        
        # Log query embedding info
        logger.info(f"[{robot_id}] 🔍 Query embedding shape: {query_embedding.shape}, norm: {np.linalg.norm(query_embedding):.6f}")
        
        # Search for more candidates to ensure we don't miss good matches
        search_k = min(len(image_paths), 7)  # Get all available images or max 7
        D, I = image_index.search(query_embedding, search_k)
        search_duration = time.perf_counter() - search_start
        
        # Enhanced logging for search results
        logger.info(f"[{robot_id}] 🎯 FAISS Search Results:")
        logger.info(f"[{robot_id}] 📊 Query: '{user_query[:50]}{'...' if len(user_query) > 50 else ''}'")
        logger.info(f"[{robot_id}] 📈 Index type: {type(image_index).__name__} (metric: {'Inner Product' if hasattr(image_index, 'metric_type') and image_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2 Distance'})")
        
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx < len(image_paths):
                image_name = os.path.basename(image_paths[idx]) if idx < len(image_paths) else f"index_{idx}"
                # Determine similarity interpretation based on index type
                if hasattr(image_index, 'metric_type') and image_index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    similarity_desc = f"cosine similarity: {score:.6f} ({'Very Similar' if score > 0.8 else 'Similar' if score > 0.5 else 'Somewhat Related' if score > 0.2 else 'Unrelated'})"
                else:
                    similarity_desc = f"L2 distance: {score:.6f} ({'Very Similar' if score < 0.5 else 'Similar' if score < 1.0 else 'Somewhat Related' if score < 1.5 else 'Unrelated'})"
                
                logger.info(f"[{robot_id}] 🏆 Rank {rank + 1}: {image_name} - {similarity_desc}")
            else:
                logger.warning(f"[{robot_id}] ⚠️ Invalid index {idx} (out of range: {len(image_paths)})")
        
        processing_duration = time.perf_counter() - processing_start
        logger.info(f"[{robot_id}] ⏱️ Search completed in {search_duration:.3f}s - Top result: {('cosine=' if hasattr(image_index, 'metric_type') and image_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'distance=')}{D[0][0]:.6f}")
        logger.info(f"[{robot_id}] 🧠 Processing breakdown - CLIP init: %.3fs, Translation: %.3fs, Embedding: %.3fs, Search: %.3fs", 
                   clip_init_duration, translation_duration, embedding_duration, search_duration)

        # Process results with enhanced semantic matching
        result_processing_start = time.perf_counter()
        closest_image_path = None
        
        # Get descriptions for semantic comparison
        descriptions = metadata.get("descriptions", [])
        
        # Define similarity thresholds for meaningful matches
        MIN_SIMILARITY_THRESHOLD = 0.05  # Very low threshold since we'll do additional semantic filtering
        MIN_DISTANCE_THRESHOLD = 2.5     # Higher threshold for L2 distance
        
        # Also do direct text similarity matching against descriptions
        from sentence_transformers import SentenceTransformer
        try:
            # Use a lightweight sentence transformer for description matching
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            query_text_embedding = sentence_model.encode([translated_text])
            description_embeddings = sentence_model.encode(descriptions)
            
            # Calculate semantic similarity with descriptions
            from sklearn.metrics.pairwise import cosine_similarity
            text_similarities = cosine_similarity(query_text_embedding, description_embeddings)[0]
            
            logger.info(f"[{robot_id}] 📝 Text similarity scores:")
            for i, (desc, sim_score) in enumerate(zip(descriptions, text_similarities)):
                logger.info(f"[{robot_id}]   {i}: '{desc}' - text similarity: {sim_score:.4f}")
        except Exception as e:
            logger.warning(f"[{robot_id}] Could not compute text similarities: {e}")
            text_similarities = None
        
        # First, check if there's a clear text similarity winner (bypass FAISS if needed)
        if text_similarities is not None:
            max_text_score = max(text_similarities)
            max_text_idx = text_similarities.argmax()
            
            logger.info(f"[{robot_id}] 🎯 Best text match: idx={max_text_idx}, score={max_text_score:.4f}")
            logger.info(f"[{robot_id}]   Description: '{descriptions[max_text_idx]}'")
            
            # If text similarity is very strong, use it directly
            if max_text_score >= 0.5:  # Very strong semantic match
                closest_image_path = image_paths[max_text_idx]
                image_name = os.path.basename(closest_image_path)
                
                logger.info(f"[{robot_id}] 🎯 SELECTED IMAGE (Strong Text Match - Bypassing FAISS): {image_name}")
                logger.info(f"[{robot_id}] 📏 Match Quality: EXCELLENT")
                logger.info(f"[{robot_id}] 📊 Text Similarity: {max_text_score:.4f} (Primary)")
                logger.info(f"[{robot_id}] 📝 Description: '{descriptions[max_text_idx]}'")
                logger.info(f"[{robot_id}] 🔗 Image Path: {closest_image_path}")
                
                set_closest_image_path(robot_id, closest_image_path)
                
                # Skip to the end
                result_processing_duration = time.perf_counter() - result_processing_start
            else:
                # Proceed with combined FAISS + text approach
                closest_image_path = None
        else:
            closest_image_path = None
        
        # Only do combined scoring if we didn't already select based on text similarity
        if closest_image_path is None:
            # Combine FAISS results with text similarity
            best_candidates = []
            for rank, idx in enumerate(I[0]):
                if idx < len(image_paths) and idx < len(descriptions):
                    faiss_score = D[0][rank]
                    text_sim_score = text_similarities[idx] if text_similarities is not None else 0.0
                    
                    # Log detailed matching info
                    logger.info(f"[{robot_id}] 🔍 Candidate {rank}: idx={idx}, FAISS={faiss_score:.4f}, Text={text_sim_score:.4f}")
                    logger.info(f"[{robot_id}]   Image: {image_paths[idx]}")
                    logger.info(f"[{robot_id}]   Description: '{descriptions[idx]}'")
                    
                    # Since text similarity is more reliable for semantic matching, weight it higher
                    # But check if this is actually a meaningful text match
                    if text_sim_score >= 0.4:  # Strong text match
                        combined_score = 0.2 * faiss_score + 0.8 * text_sim_score
                    elif text_sim_score >= 0.2:  # Moderate text match
                        combined_score = 0.4 * faiss_score + 0.6 * text_sim_score
                    else:  # Weak text match, rely more on FAISS
                        combined_score = 0.7 * faiss_score + 0.3 * text_sim_score
                    
                    best_candidates.append({
                        'idx': idx,
                        'faiss_score': faiss_score,
                        'text_score': text_sim_score,
                        'combined_score': combined_score,
                        'path': image_paths[idx],
                        'description': descriptions[idx]
                    })
        
            # Sort by combined score (higher is better)
            best_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Select the best match - prioritize text similarity since it's more reliable
            logger.info(f"[{robot_id}] 🏆 Ranking candidates by combined score:")
            for i, candidate in enumerate(best_candidates[:3]):  # Show top 3
                logger.info(f"[{robot_id}]   {i+1}. {os.path.basename(candidate['path'])} - Combined: {candidate['combined_score']:.4f} (FAISS: {candidate['faiss_score']:.4f}, Text: {candidate['text_score']:.4f})")
            
            for candidate in best_candidates:
                faiss_score = candidate['faiss_score']
                text_score = candidate['text_score']
                combined_score = candidate['combined_score']
                
                # Prioritize text similarity - if text similarity is strong, accept it
                if text_score >= 0.4:  # Strong semantic match
                    closest_image_path = candidate['path']
                    image_name = os.path.basename(closest_image_path)
                    
                    quality = "EXCELLENT" if text_score > 0.6 else "GOOD"
                    
                    logger.info(f"[{robot_id}] 🎯 SELECTED IMAGE (Strong Text Match): {image_name}")
                    logger.info(f"[{robot_id}] 📏 Match Quality: {quality}")
                    logger.info(f"[{robot_id}] 📊 Text Similarity: {text_score:.4f} (Primary), FAISS: {faiss_score:.4f} (Secondary)")
                    logger.info(f"[{robot_id}] 📝 Description: '{candidate['description']}'")
                    logger.info(f"[{robot_id}] 🔗 Image Path: {closest_image_path}")
                    
                    set_closest_image_path(robot_id, closest_image_path)
                    break
                
                # If no strong text match, check combined criteria
                elif text_score >= 0.2 and combined_score >= 0.3:  # Moderate match
                    closest_image_path = candidate['path']
                    image_name = os.path.basename(closest_image_path)
                    
                    quality = "FAIR"
                    
                    logger.info(f"[{robot_id}] 🎯 SELECTED IMAGE (Moderate Match): {image_name}")
                    logger.info(f"[{robot_id}] 📏 Match Quality: {quality}")
                    logger.info(f"[{robot_id}] 📊 Combined Score: {combined_score:.4f} (Text: {text_score:.4f}, FAISS: {faiss_score:.4f})")
                    logger.info(f"[{robot_id}] 📝 Description: '{candidate['description']}'")
                    logger.info(f"[{robot_id}] 🔗 Image Path: {closest_image_path}")
                    
                    set_closest_image_path(robot_id, closest_image_path)
                    break
                else:
                    logger.info(f"[{robot_id}] ❌ Rejecting weak match: {os.path.basename(candidate['path'])} - Text: {text_score:.4f}, Combined: {combined_score:.4f}")
            
            if not best_candidates:
                logger.error(f"[{robot_id}] ❌ No candidates found in search results")
        
        if closest_image_path is None:
            logger.warning(f"[{robot_id}] ⚠️ No relevant image found above similarity threshold for query: '{user_query[:50]}...'")
            logger.info(f"[{robot_id}] 💡 Consider: Query might be too general or unrelated to available educational content")
            
            # Check if this is a greeting or non-educational query
            greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                               'how are you', 'what\'s up', 'greetings', 'yourself', 'who are you', 
                               'tell me about yourself', 'introduce yourself']
            
            query_lower = user_query.lower()
            is_greeting = any(keyword in query_lower for keyword in greeting_keywords)
            
            if is_greeting:
                logger.info(f"[{robot_id}] 🤝 Detected greeting/personal query - no image needed")
                # For greetings, return None to indicate no image should be shown
                closest_image_path = None
            else:
                # For educational queries, we might want to return a default educational image
                # but for now, we'll just return None to avoid showing irrelevant content
                logger.info(f"[{robot_id}] 📚 Educational query with no matching image - returning None")
                closest_image_path = None
            
        result_processing_duration = time.perf_counter() - result_processing_start

        # Calculate total retrieval time
        total_retrieval_duration = time.perf_counter() - retrieval_start
        
        # Prepare timing data for analysis
        timing_data = {
            "total": total_retrieval_duration,
            "clip_init": clip_init_duration,
            "translation": translation_duration,
            "embedding": embedding_duration,
            "search": search_duration,
            "result_processing": result_processing_duration
        }
        
        if LOCAL_MODE:
            timing_data["loading"] = loading_duration
            logger.info(f"[{robot_id}] ⏱️  Image retrieval completed in %.3fs - Breakdown: Loading=%.3fs, Processing=%.3fs, Result=%.3fs", 
                       total_retrieval_duration, loading_duration, processing_duration, result_processing_duration)
        else:
            timing_data["download"] = download_duration
            logger.info(f"[{robot_id}] ⏱️  Image retrieval completed in %.3fs - Breakdown: Download=%.3fs, Processing=%.3fs, Result=%.3fs", 
                       total_retrieval_duration, download_duration, processing_duration, result_processing_duration)
        
        # Performance analysis
        analyze_image_retrieval_performance(timing_data, robot_id, LOCAL_MODE)
        
        if total_retrieval_duration > 5.0:  # More than 5 seconds
            logger.warning(f"[{robot_id}] ⚠️  Image retrieval is taking longer than expected: %.3fs", total_retrieval_duration)
        
        if closest_image_path:
            logger.info(f"[{robot_id}] ✅ Image retrieval successful: {closest_image_path}")
        else:
            logger.info(f"[{robot_id}] ✅ Image retrieval completed: No relevant image found for query")
        return closest_image_path

    except Exception as e:
        total_duration = time.perf_counter() - retrieval_start
        logger.error(f"[{robot_id}] ❌ Image retrieval failed in %.3fs: {e}", total_duration, exc_info=True)
        return None