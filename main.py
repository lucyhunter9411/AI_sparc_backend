from dotenv import load_dotenv
from pathlib import Path
# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, HTTPException, Form, WebSocket, WebSocketDisconnect, UploadFile, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict
import logging      
import shutil
import os 
import time
import json
import base64
from pydantic import BaseModel
from Schema import QnA, Topic
from bson import ObjectId
import wave
from bson import ObjectId
from app.api.deps import get_db
from app.utils.audio import generate_audio_stream
import app.services.llm as llm
from app.services.vision_service import  get_data
from app.services.shared_data import get_lecture_states, set_language_selected, get_language_selected, set_saveConv
from app.websockets.connection_manager import ConnectionManager
from app.websockets.lecture import router as lecture_router
from app.websockets.before_lecture import router as before_lecture_router
from app.vector_db.vectorDB_generation_ini import create_vector_db  
from app.vector_db.vectorDB_generation_update import process_pdf_and_create_or_update_vector_db
from app.utils.rooms import get_rooms
from app.utils.TV_app import sign_in

import subprocess
import sys
from azure.storage.blob import BlobServiceClient

# --- add just above the FastAPI() call --------------
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import asyncio
from PIL import Image
from Schema import DeviceList  # Add this import statement


from app.services.llm_service import (
    predict,                 # call the active model
    set_default_llm,         # change via /selectModel endpoint
    get_default_llm_name,    # read current name
)
selectedModelName = get_default_llm_name()

from app.core.config import get_settings
from app.core.logging import setup_logging
settings = get_settings()
setup_logging(level="INFO")        # single call replaces logging.basicConfig
logger = logging.getLogger(__name__)  # module-specific logger

TESTING: bool = os.getenv("TESTING", "0") == "1"  # added for pytest

from bson import ObjectId
from app.core.database import mongo_db 

# make sure this is at module scope, before your @app.on_event
custom_prompt_template = ""

DB_TEXT_FAISS_PATH = "app/vector_db/vectorstore/text_faiss"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
faiss_text_db = None      # will be initialised once in lifespan

# Azure Blob config
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "pdf-images"

# Initialize Azure Blob client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

manager = ConnectionManager() 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application-wide startup / shutdown lifecycle hook.

    * Loads the custom prompt template from Cosmos Mongo.
    * Loads the FAISS vector store into memory.
    * Publishes a ConnectionManager instance at `app.state.conn_mgr`
      so any request handler can `request.app.state.conn_mgr`.
    """
    global faiss_text_db, custom_prompt_template

    # ───── expose ConnectionManager early ─────
    app.state.conn_mgr = manager

    # ───── perform existing startup work ─────
    try:
        async with mongo_db() as db:
            latest_prompt = db.prompt.find_one(
                {"name": "system_prompt"}
            )
            if latest_prompt:
                custom_prompt_template = latest_prompt["prompt"]
                logger.info("✅ Loaded custom prompt template on startup.")
            else:
                logger.info("ℹ️  No custom prompt found in the database.")

        faiss_text_db = FAISS.load_local(
            DB_TEXT_FAISS_PATH,
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            allow_dangerous_deserialization=True,
        )
        logger.info("✅ FAISS text index ready (%s).", DB_TEXT_FAISS_PATH)
        if faiss_text_db is None:
                logger.error("❌ FAISS text index is None after loading.")
        else:
            logger.info("✅ FAISS text index successfully loaded.")

    except Exception:
        logger.exception("❌ Error during startup lifespan")

    # ───────────── application runs ─────────────
    yield

    # ───────────── graceful shutdown ────────────
    # (Nothing to tear down yet – FAISS is memory-mapped
    #  and ConnectionManager has no external resources.)
    if hasattr(app.state, "conn_mgr"):
        del app.state.conn_mgr


app = FastAPI(lifespan=lifespan)

# Get settings for CORS configuration
settings = get_settings()

# Build CORS origins list
allow_origins = ["http://localhost:3000"]
if settings.frontend_url:
    allow_origins.append(settings.frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(lecture_router)
app.include_router(before_lecture_router)

selectedSaveConv = True
contents = []
time_list = []

@app.get("/health", tags=["utils"])
async def health() -> dict[str, str]:
    """CI smoke-test endpoint."""
    return {"status": "ok"}

# Update the structure of lecture_states to include connectrobot
lecture_states: Dict[str, Dict[str, Dict[str, Dict]]] = {}
isSpeak = False
text = ""
data_test = {}
lecture_state_test = {}
connected_clients = {}
connected_audio_clients = {}
local_time_set = {}
robot_id_before = None

def convert_audio_to_bytes(robot_id):
    audio_path = f"{robot_id}received_audio.wav"
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        return audio_bytes
    except Exception as e:
        logger.exception("❌ Error reading audio file:")
        return None

def save_audio_to_file(audio_bytes, robot_id):
    audio_path = f"{robot_id}received_audio.wav"
    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
    logger.info(f"✅ Audio saved to {audio_path}")

@app.get("/lectures/")
async def getLectures(db=Depends(get_db)):
    allLectures = list(db.lectures.find().sort("_id", 1))
    for lecture in allLectures:
        lecture["_id"] = str(lecture["_id"])  # Convert ObjectId to string

    return {"lecture": allLectures}  # Return all lectures

@app.get("/topics/")
async def getTopics(db=Depends(get_db)):
    allTopics = list(db.topics.find().sort("_id", 1))
    for topic in allTopics:
        topic["_id"] = str(topic["_id"])
    return {"topic": allTopics}


@app.post("/selectModel/")
async def selectModel(modelName: str = Form(...)):
    global selectedModelName
    try:
        set_default_llm(modelName)
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid model name")
    selectedModelName = modelName
    return {"selected_model": modelName}

@app.post("/saveConv/{robot_id}")
async def saveConv(robot_id: str, saveConv: str = Form(...)):
    global selectedSaveConv
    selectedSaveConv = saveConv
    set_saveConv(robot_id, saveConv)
    logger.info(f"save_conversation: {selectedSaveConv}")

@app.post("/selectLanguage/{lecture_id}/{robot_id}")
async def selectLanguage(lecture_id: str, robot_id: str, languageName: str = Form(...)):
    language_selected = get_language_selected()
    # Ensure that the lecture_id exists in language_selected
    if lecture_id not in language_selected:
        language_selected[lecture_id] = {}  # Initialize the dictionary for the lecture_id
    
    # Ensure that the robot_id exists in the nested dictionary
    if robot_id not in language_selected[lecture_id]:
        language_selected[lecture_id][robot_id] = {
            "selectedLanguageName": languageName
        }
    else:
        # Update the selected language if the robot_id already exists
        language_selected[lecture_id][robot_id]["selectedLanguageName"] = languageName

    logger.info(f"Selected Language for {lecture_id}: {languageName}")
    set_language_selected(language_selected)
    return {"message": "Language updated successfully", "selectedLanguageName": languageName}

@app.post("/changeLanguage/{lecture_id}/{robot_id}")
async def change_language(lecture_id: str, robot_id: str, languageName: str = Form(...)):
    lecture_states = get_lecture_states()
    if lecture_id not in lecture_states:
        return {"message": "Lecture not started yet."}
    # Update the language for all active sessions
    for session_id in lecture_states[lecture_id][robot_id]["sessions"]:
        lecture_states[lecture_id][robot_id]["sessions"][session_id]["selectedLanguageName"] = languageName

    # logger.info(f"Language for lecture {lecture_id} changed to {languageName}")
    return {"message": "Language updated successfully", "selectedLanguageName": languageName}

chat_histories: Dict[str, List[str]] = {}
MAX_HISTORY_LENGTH = 5

def reset_session_state(session_id: str, lecture_id: str):
    global contents, time_list
    previous_language = lecture_states.get(lecture_id, {}).get("selectedLanguageName", "English")
    
    return {
        "is_active": True,
        "question_time_start": None,
        "last_message_time": None,
        "contents": contents,
        "time_list": time_list,
        "start_time": None,
        "question_active": False,
        "websocket": None, 
        "selectedLanguageName": previous_language  # Preserve language
    }


import re
def sanitize_text(text: str) -> str:
    """Removes potentially flagged words or symbols from user input."""
    forbidden_words = ["hack", "kill", "violence", "explosive", "attack", "threat"]
    for word in forbidden_words:
        text = re.sub(rf"\b{word}\b", "****", text, flags=re.IGNORECASE)
    return text

# Global variable to store selected users as pairs of (robot_id, selected_user)
stored_users = []

async def generate_and_send_ai_response(
        websocket: WebSocket, 
        lecture_state: Dict, 
        retrieve_data: str, 
        remaining_time: int, db=Depends(get_db)):
    start = time.time()
    selected_language = lecture_state.get("selectedLanguageName", "English")
    if remaining_time == 0:
        return
    prompt_for_question_time = f"""
       You are an AI assistant summarizing the following information for a young elementary school student. 
       The summary should be **simple**, **concise**, and **easy to understand**. You need to explain the context within {remaining_time}. 
       Focus on the **most important points** and keep the explanation clear, so a young child can follow it easily. 
       Here's the information to summarize:

        **CONTEXT:** 
        {retrieve_data}

        As a Text-To-Speech Model, I am using azure TTS model. 
        So generate the text translated into {selected_language} language to read for {remaining_time} seconds.
        Keep your explanation clear and to the point!
        And also, you must summarize and expand, don't generate that feel free to ask...
        I only need neccessary sentences.
    """
    
    prompt_template = PromptTemplate(input_variables=["retrieve_data", "remaining_time", "selected_language"], template=prompt_for_question_time)
    formatted_prompt = prompt_template.format(
        retrieve_data=retrieve_data,
        remaining_time=remaining_time,
        selected_language=selected_language
    )
    
    # model = llm.llm_models[selectedModelName]
    result = await predict(formatted_prompt)
    
    textTime = time.time()
    # logger.info("Text Generation Time:", int(textTime - start))
    audio_stream = generate_audio_stream(result, selected_language)
    audio_stream.seek(0)
    audio_base64 = base64.b64encode(audio_stream.read()).decode("utf-8")
    
    await websocket.send_text(json.dumps({"text": result, "audio": audio_base64, "type": "model"}))

    quesAndAnswer = {
        "question" : "automatic-generation",
        "answer" : result,
        "model" : selectedModelName,
        "prompt" : prompt_for_question_time
    }

    try:
        # Insert QnA document into the collection
        qna_document = QnA(**quesAndAnswer)  # Convert to Pydantic model
        db.qna.insert_one(qna_document.dict(exclude_unset=True))  # Insert into MongoDB
    except Exception as e:
        logger.error(f"❌ Error inserting QnA: {e}")

@app.get("/prompt/")
async def promptUpdate(db=Depends(get_db)):
    latest_prompt = db.prompt.find_one(sort=[("_id", -1)])
    logger.info("latest_prompt: %s", latest_prompt)
    return {"prompt" : latest_prompt.get("prompt")}

@app.post("/prompt/")
async def promptUpdata(prompt: str = Form(...), db=Depends(get_db)):
    global custom_prompt_template
    latest_prompt = db.prompt.find().sort([("_id", -1)]).limit(1)
    custom_prompt_template = prompt
    result = db.prompt.update_one(
        {"_id": latest_prompt[0]["_id"]},
        {"$set": {"prompt": prompt}}
    )

@app.put("/qna/update/")
async def update_qna(qna: QnA, db=Depends(get_db)):
    # logger.info(qna)
    try:
        # Find the most recent QnA (the last inserted one)
        latest_qna = db.qna.find().sort([("_id", -1)]).limit(1)
        # logger.info(latest_qna)
        if not latest_qna:
            raise HTTPException(status_code=404, detail="No QnA found to update")

        # Update the most recent QnA with the new answer
        result = db.qna.update_one(
            {"_id": latest_qna[0]["_id"]},
            {"$set": {"answer": qna.answer}}  # Update only the answer field
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="No update performed.")

        return {"response":qna.answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating QnA: {e}")


class TTSRequest(BaseModel):
    text: str
    lang: str

@app.post("/gtts/")
async def gtts(request: Request):
    body = await request.json()
    text = body.get("text")
    lang = body.get("lang", "en")  # Default to English if language is missing

    # Map languages properly
    lang_map = {
        "en": "English",
        "st": "English",
        "hi": "Hindi",
        "te": "Telugu"
    }

    selected_language = lang_map.get(lang, "English")
    audio_stream = generate_audio_stream(text, selected_language)
    audio_stream.seek(0)
    audio_base64 = base64.b64encode(audio_stream.read()).decode("utf-8")
  
    return JSONResponse({"audio": audio_base64})

UPLOAD_DIR = "static/image/"

os.makedirs(UPLOAD_DIR, exist_ok=True)

generateTextPrompt = """ 
    You are a teacher helping students understand complex concepts in a simple and engaging way. 
    Please transform {text} into a version that is clear, easy to follow, and suitable for a young or beginner audience. 
    The goal is to make the content sound natural, approachable, and interesting for students while maintaining its original meaning.
    I don't need any statements, explanations, pronounciations and approaches.
    PLease give me transformed result.
"""

@app.get("/promptGenerate/")
async def promptUpdate():
    return {"prompt" : generateTextPrompt}

@app.post("/promptGenerate/")
async def promptUpdata(prompt: str = Form(...)):
    global generateTextPrompt
    generateTextPrompt = prompt

@app.post("/generateText/")
async def generateText(text: str = Form(...)):

    #AI generation
    global generateText
    prompt_template = PromptTemplate(input_variables=["text"], template=generateTextPrompt)
    formatted_prompt = prompt_template.format(
        text=text,
    )
    model =llm.llm_models[selectedModelName]
    englishData = model.predict(formatted_prompt)

    # Hindi Generation
    generateTextPrompt_Hindi = """Please translate {text} into Hindi language. I don't need any statements, explanations, pronounciations and approaches. PLease give me translated result. """
    prompt_template_Hindi = PromptTemplate(input_variables=["text"], template=generateTextPrompt_Hindi)
    formatted_prompt_hindi = prompt_template_Hindi.format(
        text=englishData,
    )
    hindiData = model.predict(formatted_prompt_hindi)

    # Telugu Generation
    generateTextPrompt_Telugu = """lease translate {text} into Telugu language. I don't need any statements, explanations, pronounciations and approaches. PLease give me translated result. """
    prompt_template_Telugu = PromptTemplate(input_variables=["text"], template=generateTextPrompt_Telugu)
    formatted_prompt_Telugu = prompt_template_Telugu.format(
        text=englishData,
    )
    teluguData = model.predict(formatted_prompt_Telugu)

    return {"English": englishData, "Hindi": hindiData, "Telugu":teluguData}

@app.post("/upload/")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Save the image to the server
        image_path = os.path.join(UPLOAD_DIR, image.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        return JSONResponse(content={"imageUrl": image.filename})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/imageUpdate/")
async def imageUpdate(topic: Topic, db=Depends(get_db)):
    # Ensure that the id exists
    if topic.id is None:
        raise HTTPException(status_code=400, detail="Topic ID is required.")
    
    # Convert the string id to ObjectId if it's a string
    try:
        topic_id = ObjectId(topic.id)  # Convert the id to ObjectId for MongoDB
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Topic ID format.")
    
    # Convert Topic to dictionary (excluding _id)
    topic_dict = topic.dict(exclude_unset=True, exclude={"_id"})
    
    # Perform the update operation
    result = db.topics.update_one(
        {"_id": topic_id},  # Use the converted ObjectId in the query
        {"$set": topic_dict}  # Use the dict representation for update
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Topic not found or no changes made")
    
    return {"message": "Topic updated successfully!"}

@app.post("/newContent/")
async def newContent(topic: Topic, db=Depends(get_db)):
    global generateText
    if topic.id is None:
        raise HTTPException(status_code=400, detail="Topic ID is required.")
    try:
        topic_id = ObjectId(topic.id)  # Convert the id to ObjectId for MongoDB
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Topic ID format.")
    
    
    
    text = topic.content[-1].text
    # logger.info("TEXT======================",text)
    prompt_template = PromptTemplate(input_variables=["text"], template=generateTextPrompt)
    formatted_prompt = prompt_template.format(
        text=text,
    )

  
    englishData = await predict(formatted_prompt)

    # Hindi Generation
    generateTextPrompt_Hindi = """ Please translate {text} into Hindi language. I don't need any statements, explanations, pronounciations and approaches. PLease give me translated result. """
    prompt_template_Hindi = PromptTemplate(input_variables=["text"], template=generateTextPrompt_Hindi)
    formatted_prompt_hindi = prompt_template_Hindi.format(
        text=englishData,
    )
    hindiData = await predict(formatted_prompt_hindi)

    # Telugu Generation
    generateTextPrompt_Telugu =  """ Please translate {text} into Telugu language. I don't need any statements, explanations, pronounciations and approaches. PLease give me translated result. """
    prompt_template_Telugu = PromptTemplate(input_variables=["text"], template=generateTextPrompt_Telugu)
    formatted_prompt_Telugu = prompt_template_Telugu.format(
        text=englishData,
    )
    teluguData = await predict(formatted_prompt_Telugu)
    topic.content[-1].EnglishText = englishData
    topic.content[-1].HindiText = hindiData
    topic.content[-1].TeluguText = teluguData
    topic_dict = topic.dict(exclude_unset=True, exclude={"_id"})
    result = db.topics.update_one(
        {"_id": topic_id},  # Use the converted ObjectId in the query
        {"$set": topic_dict}  # Use the dict representation for update
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Topic not found or no changes made")
    
    return {"message": topic}

@app.post("/addTopic/")
async def addTopic(lecture_id: str = Form(...), title: str = Form(...), qna_time: int = Form(...), db=Depends(get_db)):
    topic_data = {
        "lecture_id": lecture_id,
        "title": title,
        "qna_time": qna_time,
        "content": []  # Empty list for content
    }

    # Insert into 
    result = db.topics.insert_one(topic_data)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Failed to insert topic")

    allTopics = list(db.topics.find().sort("_id", 1))
    for topic in allTopics:
        topic["_id"] = str(topic["_id"])
    return {"topic": allTopics}

@app.post("/topicsUpdate/")
async def topicsUpdate(topic: Topic, db=Depends(get_db)):

    if isinstance(topic.id, str):
        try:
            topic_id = ObjectId(topic.id)  
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid ObjectId format: {topic.id}")
    else:
        topic_id = topic.id  
    
    result = db.topics.update_one(
        {"_id": topic_id},
        {"$set": topic.dict(exclude_unset=True)} 
    )
    
    if result.modified_count == 0:
        logger.info(f"No changes made for topic with _id: {topic.id}")
    else:
        logger.info(f"Updated topic with _id: {topic.id}")
    
    return {"message": "Topics updated successfully!"}

@app.post("/deleteTopic/")
async def deleteTopic(topicID: str = Form(...), db=Depends(get_db)):
    # Ensure the topicID is a valid ObjectId (MongoDB's ID format)
    try:
        # Convert the string to an ObjectId
        object_id = ObjectId(topicID)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid topicID format")

    # Perform the deletion operation
    result =db.topics.delete_one({"_id": object_id})

    if result.deleted_count == 0:
        # If no document was deleted, return an error
        raise HTTPException(status_code=404, detail="Topic not found")

    allTopics = list(db.topics.find().sort("_id", 1))
    for topic in allTopics:
        topic["_id"] = str(topic["_id"])
    return {"topic": allTopics}

@app.post("/addLecture/")
async def addLecture(title: str = Form(...), db=Depends(get_db)):
    # Insert lecture data
    lecture_data = {"title": title}
    result = db.lectures.insert_one(lecture_data)
    if not result.inserted_id:
        raise HTTPException(status_code=500, detail="Failed to insert lecture")

    # Fetch all lectures and return with ObjectId as string
    allLectures = list(db.lectures.find().sort("_id", 1))
    for lecture in allLectures:
        lecture["_id"] = str(lecture["_id"])  # Convert ObjectId to string

    return {"lecture": allLectures}  # Return all lectures

class VisionData(BaseModel):
    handup_result: list
    face_recognition_result: list
    robot_id: str
    image_name: str  # The name of the image file
    image: str  # The image data in bytes
    detect_user: list
    local_time_vision: int

@app.post("/vision/getData/")  # rename /vision/update/
async def get_data_endpoint(vision_data: VisionData):  # rename visionUpdate
    logger.info("Enter 'vision/getData'")
    await get_data(vision_data)  # Call the new function


@app.post("/deleteLecture/")
async def deleteLecture(lectureID: str = Form(...), db=Depends(get_db)):
    # Ensure the topicID is a valid ObjectId (MongoDB's ID format)
    try:
        # Convert the string to an ObjectId
        object_id = ObjectId(lectureID)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid topicID format")

    # Perform the deletion operation
    result =db.lectures.delete_one({"_id": object_id})

    if result.deleted_count == 0:
        # If no document was deleted, return an error
        raise HTTPException(status_code=404, detail="Topic not found")

    allLectures = list(db.lectures.find().sort("_id", 1))
    for lecture in allLectures:
        lecture["_id"] = str(lecture["_id"])
    return {"lecture": allLectures}

# Define the absolute path for uploads
UPLOAD_FOLDER_FAISS = os.getenv("UPLOAD_FOLDER_FAISS", "uploads")
IMAGE_DIR = os.getenv("IMAGE_DIR", "app/vector_db/images")
os.makedirs(UPLOAD_FOLDER_FAISS, exist_ok=True)

# @app.post("/create-vector-db/v0/")
# async def create_vector_db_v0_endpoint(file_name: str = Form(...)):
#     try:
#         file_location = os.path.join(UPLOAD_FOLDER_FAISS, file_name)
#         create_vector_db(file_location)
#         return {"status": "Vector DB created successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/create-vector-db/v0/")
async def create_vector_db_v0_endpoint(file_name: str = Form(...)):
    try:
        file_location = os.path.join(UPLOAD_FOLDER_FAISS, file_name)

        # Run with the current Python interpreter path
        subprocess.Popen([sys.executable, "app/vector_db/vectorDB_generation_ini.py", file_location])

        return {
            "status": "processing",
            "message": f"Embedding for '{file_name}' has started in background."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        
# @app.post("/create-vector-db/v1/")
# async def create_vector_db_v1_endpoint(file_name: str = Form(...)):
#     try:
#         file_location = os.path.join(UPLOAD_FOLDER_FAISS, file_name)
#         process_pdf_and_create_or_update_vector_db(file_location)
#         return {"status": "Vector DB created successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-vector-db/v1/")
async def create_vector_db_v1_endpoint(file_name: str = Form(...)):
    try:
        file_location = os.path.join(UPLOAD_FOLDER_FAISS, file_name)

        # Run with the current Python interpreter path
        subprocess.Popen([sys.executable, "app/vector_db/vectorDB_generation_update.py", file_location])

        return {
            "status": "processing",
            "message": f"Embedding for '{file_name}' has started in background."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/upload/file/")
# async def upload_file(file: UploadFile = File(...)):
#     # Log the file received
#     if file is None:
#         raise HTTPException(status_code=422, detail="No file provided.")
    
#     try:
#         file_location = os.path.join(UPLOAD_FOLDER_FAISS, file.filename)
#         with open(file_location, "wb") as f:
#             f.write(await file.read())
#         return JSONResponse(status_code=200, content={"message": "File uploaded successfully!", "file_name": file.filename})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"message": f"File upload failed: {str(e)}"})

@app.post("/upload/file/")
async def upload_file(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=422, detail="No file provided.")

    try:
        # Read file bytes
        contents = await file.read()
        blob_name = file.filename

        # Upload to Azure Blob Storage
        container_client.upload_blob(name=blob_name, data=contents, overwrite=True)

        # Construct public blob URL
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{blob_name}"

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully!",
                "file_name": file.filename,
                "blob_url": blob_url
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"File upload failed: {str(e)}"})

@app.post("/add_or_update/room/")
async def add_or_update_room(
    robot_id: str = Form(...),
    device_id: str = Form(...),
    room_name: str = Form(...),
    db=Depends(get_db)
):
    # Create a new DeviceList object
    new_device = DeviceList(robot_id=robot_id, device_id=device_id, room_name=room_name)

    try:
        # Check if the robot_id already exists in the database
        existing_entry = db.devices.find_one({"robot_id": robot_id})

        if existing_entry:
            # Check if the device_id already exists in the device list
            device_exists = any(device['device_id'] == device_id for device in existing_entry['device'])

            if device_exists:
                # Update the device name if the device_id exists
                db.devices.update_one(
                    {"robot_id": robot_id, "device.device_id": device_id},
                    {"$set": {"device.$.room_name": room_name}}
                )
                return {"message": "Device name updated for existing device_id"}
            else:
                # Append the new device to the existing list if device_id does not exist
                db.devices.update_one(
                    {"robot_id": robot_id},
                    {"$push": {"device": new_device.dict(by_alias=True)}}
                )
                return {"message": "Device added to existing robot_id"}
        else:
            # If robot_id does not exist, create a new entry
            new_entry = {
                "robot_id": robot_id,
                "device": [new_device.dict(by_alias=True)]
            }
            db.devices.insert_one(new_entry)
            return {"message": "New robot_id created and device added"}

    except Exception as e:
        logger.error(f"Error adding device: {e}")
        raise HTTPException(status_code=500, detail="Failed to add device")
    
@app.delete("/delete/room/")
async def delete_room(
    robot_id: str = Form(...),
    device_id: str = Form(...),
    room_name: str = Form(...),
    db=Depends(get_db)
):
    try:
        # Check if the robot_id and device_id exist in the database
        existing_entry = db.devices.find_one({"robot_id": robot_id, "device.device_id": device_id})

        if existing_entry:
            # Check if the room_name matches
            device_matches = any(device['device_id'] == device_id and device['room_name'] == room_name for device in existing_entry['device'])

            if device_matches:
                # Remove the device from the list
                db.devices.update_one(
                    {"robot_id": robot_id},
                    {"$pull": {"device": {"device_id": device_id, "room_name": room_name}}}
                )
                return {"message": "Device deleted successfully"}
            else:
                return {"message": "Device name does not match"}
        else:
            return {"message": "Device not found"}

    except Exception as e:
        logger.error(f"Error deleting device: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete device")
    
@app.get("/get/rooms/")
async def get_rooms_endpoint(robot_id: str = Form(...), db=Depends(get_db)):
    return await get_rooms(robot_id, db)

@app.post("/send_request/login/")
async def handle_login_request(username: str = Form(...), password: str = Form(...)):
    try:
        auth_response = sign_in(username, password)
        access_token = auth_response.get("access_token")
        # set_access_token(access_token)
        return {"message": "Login successful", "data": auth_response}
    except HTTPException as e:
        return {"message": "Login failed", "detail": str(e)}

method = "Whisper"
@app.post("/sttMethod/")
async def sttMethod(sttMethod: str = Form(...)):    
    global method
    method = sttMethod

app.mount("/static", StaticFiles(directory="static"), name="static") 
# Mount the directory that contains the images
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)