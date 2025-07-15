from dotenv import load_dotenv
from pathlib import Path
# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import faiss
import torch
import os
import json
import logging

uploads_dir = os.getenv("UPLOAD_FOLDER_FAISS", "uploads")

def create_vector_db(uploaded_file):
    logger = logging.getLogger(__name__)
    try:
        # Initialize Models & Directories
        logger.info("Initializing models and directories...")
        text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # DB_TEXT_FAISS_PATH = "app/vector_db/vectorstore/text_faiss"
        # DB_IMAGE_FAISS_PATH = "app/vector_db/vectorstore/image_faiss"
        # IMAGE_DIR = "app/vector_db/images"
        DB_TEXT_FAISS_PATH = os.getenv("DB_TEXT_FAISS_PATH", "")
        DB_IMAGE_FAISS_PATH = os.getenv("DB_IMAGE_FAISS_PATH", "")
        IMAGE_DIR = os.getenv("IMAGE_DIR", "")
        os.makedirs(IMAGE_DIR, exist_ok=True) 


        # Embed Text from PDFs
        logger.info("Embedding text from PDFs...")
        pdf_files = [uploaded_file]
        # pdf_files = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.endswith('.pdf')]
        print(pdf_files)
        all_splits = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for pdf in pdf_files:
            if os.path.exists(pdf):
                print(f"Processing PDF: {pdf}")

                loader = PyPDFLoader(pdf)
                docs = loader.load()
                splits = text_splitter.split_documents(docs)
                all_splits.extend(splits)
            else:
                logger.error(f"PDF file not found: {pdf}")

        # Store text embeddings in FAISS
        logger.info("Storing text embeddings in FAISS...")
        text_vectorstore = FAISS.from_documents(all_splits, text_embeddings)
        text_vectorstore.save_local(DB_TEXT_FAISS_PATH)
        logger.info(f"Text embeddings saved to {DB_TEXT_FAISS_PATH}")

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

        # Save metadata (image descriptions)
        metadata_path = f"{DB_IMAGE_FAISS_PATH}.json"
        with open(metadata_path, "w") as f:
            json.dump({"image_paths": image_paths, "descriptions": image_descriptions}, f, indent=4)
        logger.info(f"Image metadata saved to {metadata_path}")

    except Exception as e:
        logger.exception("Error creating vector DB: %s", e)
        raise