from dotenv import load_dotenv
from pathlib import Path
# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import faissimport os
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

import torch
import os
import json
import logging
from azure.storage.blob import BlobServiceClient, ContentSettings
import requests
import tempfile

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

def download_pdf(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF: {pdf_url}")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

def create_vector_db(uploaded_file):
    logger = logging.getLogger(__name__)
    try:
        # Initialize Models & Directories
        logger.info("Initializing models and directories...")
        text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        DB_TEXT_FAISS_PATH = os.getenv("DB_TEXT_FAISS_PATH", "")
        DB_IMAGE_FAISS_PATH = os.getenv("DB_IMAGE_FAISS_PATH", "")
        IMAGE_DIR = os.getenv("IMAGE_DIR", "")
        os.makedirs(IMAGE_DIR, exist_ok=True) 

        # Embed Text from PDFs
        logger.info("Embedding text from PDFs...")
        pdf_files = [uploaded_file]
        all_splits = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for pdf in pdf_files:
            if pdf.startswith("http://") or pdf.startswith("https://"):
                pdf = download_pdf(pdf)

            if os.path.exists(pdf):
                logger.info(f"Processing PDF: {pdf}")

                loader = PyPDFLoader(pdf)
                docs = loader.load()
                splits = text_splitter.split_documents(docs)
                all_splits.extend(splits)
            else:
                logger.error(f"PDF file not found: {pdf}")

        if not all_splits:
            raise ValueError("No text extracted from the PDF. Please check the PDF content.")

        # Store text embeddings in FAISS
        logger.info("Storing text embeddings in FAISS...")
        text_vectorstore = FAISS.from_documents(all_splits, text_embeddings)
        text_vectorstore.save_local(DB_TEXT_FAISS_PATH)
        logger.info(f"Text embeddings saved to {DB_TEXT_FAISS_PATH}")

        # Correctly set the local path for the FAISS index
        text_faiss_path = os.path.join(tempfile.gettempdir(), "text_faiss.faiss")
        faiss.write_index(text_vectorstore.index, text_faiss_path)

        # Upload text FAISS index to blob
        text_url = upload_to_blob(text_faiss_path, "text_faiss")
        logger.info(f"Text FAISS index uploaded to {text_url}")

        # Generate embeddings for descriptions
        logger.info("Generating embeddings for image descriptions...")
        
        image_descriptions = [
            "This image is about cause of floatation and sinking. The floating or sinking of an object doesn't depend upon the weight of the object. It depends upon the density of the object and the density of the fluid. The density of an object is measured by calculating the amount of mass present in the given space. If an object’s density is lesser than water, it will float. On the other hand, if the density of an object is more than water, it will sink.",
            "This is image about dispersal of seeds by animals. Animals disperse seeds in several ways. Humans and animals eat fleshy fruits and throw away their seeds in different places. Some fruits are eaten by animals and birds. The seeds, eaten along with the fruits are not digested. These animals excrete the seeds in different places. The seeds emerge into new plants.",
            "This is image about dispersal of seeds by explosion. Some fruits scatter their seeds by exploding the seed pods. When the seeds are ripe and the pod has dried, it bursts open and the seeds are scattered. For example, pods of squirting cucumber spread their seeds by explosion. Pea and bean plants also keep their seeds in a pod. When the seeds are ripe and the pod has dried, the pod bursts open. The peas and beans are scattered.",
            "This is image about dispersal of seeds by water. Many aquatic plants that live near water have seeds that can float. They are carried by water. Plants living along streams and rivers have seeds that float downstream. Therefore, germination is possible at new sites. The size of the seed is not a factor in etermining whether a seed can float. Some very large seeds like coconut can also float. The fruits of these plants are spongy.",
            "This is image about dispersal of seeds by wind. Some seeds are small and light in weight. They have special tufts of wings or hair present around them. These structures help them get blown away with the help of winds. Seeds like dandelion have tufts of hair that help them to float in air.",
            "This is image about evaporation. The change of water into water vapour due to heating is called evaporation. The amount of water vapour present in the air is called humidity. Evaporation takes place all the time at all places.",
            "This is image about filtration. Filtration is the process in which solid particles in a liquid or gaseous fluid are removed using a filter medium that permits the fluid to pass through but retains the solid particles. Dissolved impurities are called soluble impurities.",
            "This is image about  plants that grow from leaves. Leaves of some plants like podophyllum and begonia have many buds on their edges. These buds further grow into new plants in favourable conditions.",
            "This is image about  plants that grow from roots. Sweet potato, carrot and radish have modified roots that store food. They also have buds on them, which grow into a new plant when replanted.  Some roots plants like dahlia, guava and asparagus also reproduce by their roots. These roots give rise to new plants in favourable conditions.",
            "This is image about  plants that grow from stems. Potatoes, ginger, rose, sugar cane and money plant can grow from their stem cutting. Plants like roses grow from stem cuttings. Cut the stem of the rose plant and bury it in the soil of another pot. Keep the pot under sunlight and water it regularly. After some days, the stem starts forming roots and sprouting new leaves. Potato, colocasia and ginger grow as underground stems. Buds are found on these plants at certain points, which grow into new plants under favourable conditions",
            "This is image about grasses that is one of types of seeds commonly used in agriculture. The cereals, wheat, barley, oats, maize, sorghum and rice, are all grasses. Their embryo has just one cotyledon or seed leaf. Hence, they are called monocotyledons. The embryo usually sits close to the surface of the seed and often is referred to as the 'foetus'.",
            "This is image about Legumes that is one of types of seeds commonly used in agriculture. This group (Legumes) includes peas, the various beans such as french beans, chickpeas and lentils. The embryo is normally inside the seed and includes two seed leaves (cotyledons), so they are called dicotyledons. Their food storage is contained in the cotyledons.",
            "This is image about necessary conditions for a seed to sprout. A seed needs enough air, water, sunlight and nutrients to grow into a new plant. If any condition is missing, the seed will not grow into a new plant. Some seeds need special treatment or conditions of light, temperature, moisture, etc. to germinate.",
            "This is image about Oilseeds that is one of types of seeds commonly used in agriculture. This group (Oilseeds) includes sunflower and soyabeans. These are also dicotyledons, but their food storage contains much higher levels of oil than the legumes or grasses.",
            "This is image about sedimentation and decantation. You will see that the sand particles get settled down at the bottom of the glass. This process is called sedimentation.When you pour this water into other glass, you will get pure water. This process is called decantation.",
            "This is image about sink and float. The tendency of an object to remain on the surface of water is known as floatation. Objects such as leaves, wooden sticks, paper, plastic bottles float on the surface of water. When we see something balancing on the surface of water, we say it is .  Sinking is exactly opposite to floatation. The tendency of an object to go deep down in the fluid is known as . In case the object goes under water, we say it has sunk. An iron nail, a metal object, bricks are some examples of sinking objects.",
            "This is image about solubility of Liquids in water. Some liquids can dissolve in water while some cannot. Liquids that dissolve in water or in other liquids are called miscible liquids. While the liquid that doesn't dissolve in water or in other liquids are called immiscible liquids.",
            "This is image about solubility of Solids in water. When the particles of a solid substance fill the space among the particles of liquid, it mixes with liquid. For example: Add a teaspoonful of sugar in water. Stir it well and leave it. You will get a solution of sugar in water. Sugar particles spread among the water particles.",
            "This image is stages of germination of a seed. Seeing a tiny seedling come out from a dry wrinkled seed and watching its growth and transformation is an elaborate process.",
            "This image is structure of seed. Seeds are responsible in most plants for the continuation of progeny. Seeds are special parts of the plant that are found inside the fruit of the plant. Seeds have a special structure called cotyledon that stores food inside them. This stored food is used by the plant when it is growing from the seed. A seed is made up of a seed coat and an embryo. The embryo is made up of a radicle, an embryonal axis and one (wheat, maize) or two cotyledons (gram and pea). A seed converts into a new plant when we sow it.",
            "This image is about types of seeds. Monocot: A monocot seed has only one cotyledon, that is, the seed is not divided into two parts.",
            "This image is about types of seeds. Dicot: A dicot has two cotyledons. The seed is divided into two parts.",
            "This image is Velcro. George de Mestral, a Swiss engineer and enthusiast mountaineer, was trekking in the woods with his dog in 948. When he returned home, he saw the burrs that adhered to his clothes and pondered if such a concept may be effective in commercial use. He examined a burr under a microscope and discovered that it was coated in tiny hooks that allowed it to grip onto garments and fur that brushed against it in passing. This leads to the discovery of Velcro."
        ]

        image_filenames = [
            "CAUSE OF FLOATATION AND SINKING.png",
            "Dispersal of Seeds by Animals.png",
            "Dispersal of Seeds by Explosion.png",
            "Dispersal of Seeds by Water.png",
            "Dispersal of Seeds by Wind.png",
            "EVAPORATION.png",
            "FILTRATION.png",
            "FROM LEAVES.png",
            "FROM ROOTS.png",
            "FROM STEMS.png",
            "Grasses.png",
            "Legumes.png",
            "NECESSARY CONDITIONS FOR A SEED TO SPROUT.png",
            "Oilseeds.png",
            "SEDIMENTATION AND DECANTATION.png",
            "Sink and float.png",
            "Solubility of Liquids in Water.png",
            "Solubility of Solids in Water.png",
            "STAGES OF GERMINATION OF A SEED.png",
            "Structure of Seed.png",
            "Types of Seeds_1.png",
            "Types of Seeds_2.png",
            "Velcro.png",
        ]
        image_paths = [os.path.join(IMAGE_DIR, img) for img in image_filenames]

        # Generate embeddings for descriptions
        image_desc_embeddings = []
        for description in image_descriptions:
            text_inputs = clip_processor(
                text=[description], 
                return_tensors="pt", 
                padding=True,  
                truncation=True,  
            )

            with torch.no_grad():
                embedding = clip_model.get_text_features(**text_inputs).numpy().squeeze()
            image_desc_embeddings.append(embedding)

        # Convert embeddings to NumPy array and normalize
        image_desc_embeddings = np.array(image_desc_embeddings)
        assert image_desc_embeddings.shape[1] == 512, f"Expected 512 dimensions, but got {image_desc_embeddings.shape[1]}"

        # Normalize embeddings for FAISS
        faiss.normalize_L2(image_desc_embeddings)

        # Store description embeddings in FAISS
        logger.info("Storing description embeddings in FAISS...")
        dimension = 512
        image_index = faiss.IndexFlatIP(dimension)
        image_index.add(image_desc_embeddings)

        # Save FAISS index
        faiss_path = f"{DB_IMAGE_FAISS_PATH}.faiss"
        faiss.write_index(image_index, faiss_path)
        logger.info(f"Image description embeddings saved to {faiss_path}")

        # Upload image FAISS index to blob
        image_url = upload_to_blob(faiss_path, "image_faiss")
        logger.info(f"Image FAISS index uploaded to {image_url}")

        # Save metadata (image descriptions)
        metadata_path = f"{DB_IMAGE_FAISS_PATH}.json"
        with open(metadata_path, "w") as f:
            json.dump({"image_paths": image_paths, "descriptions": image_descriptions}, f, indent=4)
        logger.info(f"Image metadata saved to {metadata_path}")

        # Upload metadata to blob
        meta_url = upload_to_blob(metadata_path, "image_faiss")
        logger.info(f"Image metadata uploaded to {meta_url}")

    except Exception as e:
        logger.exception("Error creating vector DB: %s", e)
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python vector_db_generate_ini.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    create_vector_db(pdf_path)
