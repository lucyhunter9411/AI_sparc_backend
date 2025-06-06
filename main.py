from dotenv import load_dotenv
from pathlib import Path
# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from fastapi import FastAPI, HTTPException, Form, WebSocket, WebSocketDisconnect, UploadFile, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from typing import List, Dict
import logging      
from urllib.parse import quote_plus
import shutil
import os 
import asyncio
import time
import json
import io
import base64
from pydantic import BaseModel
from Schema import QnA, Topic
from bson import ObjectId
import azure.cognitiveservices.speech as speechsdk
import wave
from transitions import Machine,State
from langdetect import detect
import openai
import speech_recognition as sr
import uuid  # Import uuid for generating unique session IDs
from bson import ObjectId
from app.api.deps import get_db
from app.api.deps import get_db
from app.utils.audio import generate_audio_stream, get_audio_length
from app.services.stt_service import transcribe_audio
from app.services.llm_service import build_chat_prompt
import app.services.llm as llm
from app.state.lecture_state_machine import LectureStateMachine
from app.services.shared_data import set_contents, set_time_list
from app.services.vision_service import  handle_vision_data, get_data
from datetime import datetime
from app.services.shared_data import get_selected_student, set_lecture_states
from app.websockets.connection_manager import ConnectionManager
from app.websockets.lecture import lecture_websocket_endpoint
from app.websockets.lecture import router as lecture_router
from app.websockets.before_lecture import router as before_lecture_router
from app.services.shared_data import get_lecture_states

# --- add just above the FastAPI() call --------------
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


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

# DB_TEXT_FAISS_PATH = "vectorstore/text_faiss"
# faiss_text_db = FAISS.load_local(
#     DB_TEXT_FAISS_PATH, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
#     allow_dangerous_deserialization=True
# )

# app = FastAPI()

from bson import ObjectId
from app.core.database import mongo_db 

# make sure this is at module scope, before your @app.on_event
custom_prompt_template = ""

DB_TEXT_FAISS_PATH = "vectorstore/text_faiss"
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
faiss_text_db = None      # will be initialised once in lifespan



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
                {"_id": ObjectId("67b72394f8b916a8d95503f6")}
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://app-ragfrontend-dev-wus-001.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(lecture_router)
app.include_router(before_lecture_router)

# model_dict = llm.llm_models
# model      = llm.get_selected_llm()   # or model_dict["GPT-4"]
# selectedModelName =  llm.get_selected_llm_name()

selectedSaveConv = True
contents = []
time_list = []

@app.get("/health", tags=["utils"])
async def health() -> dict[str, str]:
    """CI smoke-test endpoint."""
    return {"status": "ok"}

# Update the structure of lecture_states to include connectrobot
lecture_states: Dict[str, Dict[str, Dict[str, Dict]]] = {}
lecture_states = get_lecture_states()
isSpeak = False
text = ""
data_test = {}
lecture_state_test = {}
connected_clients = {}
connected_audio_clients = {}
local_time_set = {}

robot_id_before = None

@app.websocket("/ws/{lecture_id}/{connectrobot}")
async def websocket_endpoint(websocket: WebSocket, lecture_id: str, connectrobot: str = None):
    global isSpeak, text, data_test, robot_id_before
    await websocket.accept()

    session_id = str(websocket.client)

    robot_backends = {}
    robot_spoken_text = {}
    current_state_machine = None

    def update_current_state_machine(value):
        nonlocal current_state_machine
        current_state_machine = value
        logger.debug(f"current_state_machine {current_state_machine}")

    async def process_audio_message(audio_path, model, websocket, robot_id, spoken_text, with_style):
        try:
            text_result = await transcribe_audio(audio_path, model)
            logger.info(f"Transcription complete: {text_result}")

            if with_style:
                await websocket.send_text(json.dumps({"text": text_result}))
                await current_state_machine.enter_conversation(websocket, lecture_states, text_result, robot_id)
            else:
                full_text = "\n User: " + text_result
                logger.info(f"Sending to connected clients: {full_text}")
                logger.info(f"connected_audio_clients: {connected_audio_clients}")
                for client in connected_audio_clients.get(robot_id, []):
                    if client != websocket:
                        await current_state_machine.enter_conversation(client, lecture_states, full_text, robot_id)
                await websocket.send_text(json.dumps({"text": full_text}))

            if robot_id in lecture_state_test and "last_message_time" in lecture_state_test[robot_id]:
                lecture_state_test[robot_id]["last_message_time"] = time.time()

            current_state_machine.ev_init()

        except Exception as e:
            logger.exception("Error in background audio processing task:")
            await websocket.send_text(json.dumps({"error": "Audio processing failed"}))

    if lecture_id == "before":
        if lecture_id not in lecture_states:
            lecture_states[lecture_id] = {"state_machine": LectureStateMachine(lecture_id), "sessions": {}, "running": True}

        state_machine = lecture_states[lecture_id]["state_machine"]
        update_current_state_machine(state_machine)
        state_machine.ev_init()

        vision_task = asyncio.create_task(handle_vision_data(current_state_machine, robot_id_before, websocket))

        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)

                if data:
                    logger.info("Receive data from speech client successfully!")

                if data.get("type") == "ping":
                    logger.info("websocket is open!")
                    continue

                if "robot_id" in data:
                    robot_id_before = data["robot_id"]
                    if "local_time" in data:
                        local_time = data["local_time"]
                        dt = datetime.fromisoformat(local_time)
                        formatted_time = dt.strftime("%H")
                        if robot_id_before not in local_time_set:
                            local_time_set[robot_id_before] = {"local time": formatted_time}
                        else:
                            local_time_set[robot_id_before]["local time"] = formatted_time
                        logger.debug(f"local_time_set {local_time_set}")
                    if robot_id_before not in connected_clients:
                        connected_clients[robot_id_before] = []
                    if "client" in data:
                        connected_clients[robot_id_before].append(websocket)
                        logger.info("Added audio client!")
                    if "style" not in data:
                        if robot_id_before not in connected_audio_clients:
                            connected_audio_clients[robot_id_before] = []
                        if "client" in data:
                            connected_audio_clients[robot_id_before].append(websocket)

                if "backend" in data:
                    if robot_id_before:
                        robot_backends[robot_id_before] = data["backend"]

                if "spoken_text" in data:
                    if robot_id_before:
                        robot_spoken_text[robot_id_before] = data["spoken_text"]
                        text = "Assistant:" + robot_spoken_text[robot_id_before]

                if "audio" in data:
                    if robot_id_before:
                        audio_path = f"{robot_id_before}received_audio.wav"
                        if isinstance(data["audio"], str):
                            audio_bytes = base64.b64decode(data["audio"])
                        else:
                            logger.warning(f"Invalid audio data format: {type(data['audio'])}")
                            return

                        with wave.open(audio_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(audio_bytes)

                        logger.info(f"Audio saved to {audio_path}")

                        model = robot_backends.get(robot_id_before, "whisper-1")
                        with_style = "style" in data
                        spoken = robot_spoken_text.get(robot_id_before, "")

                        asyncio.create_task(process_audio_message(audio_path, model, websocket, robot_id_before, spoken, with_style))

        except WebSocketDisconnect:
            logger.error("❌ WebSocket disconnected unexpectedly: %s", session_id)
            if robot_id_before and websocket in connected_clients.get(robot_id_before, []):
                connected_clients[robot_id_before].remove(websocket)
                if not connected_clients[robot_id_before]:
                    del connected_clients[robot_id_before]
            if robot_id_before and websocket in connected_audio_clients.get(robot_id_before, []):
                connected_audio_clients[robot_id_before].remove(websocket)
                if not connected_audio_clients[robot_id_before]:
                    del connected_audio_clients[robot_id_before]
            if lecture_id in lecture_states and connectrobot in lecture_states[lecture_id]:
                lecture_states[lecture_id][connectrobot]["sessions"].pop(session_id, None)

        except KeyError:
            logger.exception("❌ KeyError - Missing expected key:")
            await websocket.send_text(json.dumps({"error": "Key error occurred."}))
        except ValueError:
            logger.exception("❌ ValueError occurred during processing:")
            await websocket.send_text(json.dumps({"error": "Value error occurred."}))
        except Exception as e:
            logger.exception(f"❌ Unexpected error with {robot_id_before}:")
            await websocket.send_text(json.dumps({"error": "Internal server error"}))
            await asyncio.sleep(2)
        finally:
            try:
                if robot_id_before and websocket in connected_clients.get(robot_id_before, []):
                    connected_clients[robot_id_before].remove(websocket)
                    if not connected_clients[robot_id_before]:
                        del connected_clients[robot_id_before]
                if robot_id_before and websocket in connected_audio_clients.get(robot_id_before, []):
                    connected_audio_clients[robot_id_before].remove(websocket)
                    if not connected_audio_clients[robot_id_before]:
                        del connected_audio_clients[robot_id_before]

                vision_task.cancel()
                try:
                    await vision_task
                except asyncio.CancelledError:
                    logger.info("Vision task cancelled cleanly.")

                # Cancel background tasks if stored in a list
                for task in getattr(websocket, "background_tasks", []):
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.info("Background task cancelled cleanly.")
            except Exception as cleanup_error:
                logger.exception("Error during WebSocket cleanup: %s", cleanup_error)


    else:
        await lecture_websocket_endpoint(websocket, lecture_id, connectrobot)

def convert_audio_to_bytes(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        return audio_bytes
    except Exception as e:
        logger.exception("❌ Error reading audio file:")
        return None

def save_audio_to_file(audio_path, audio_bytes):
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

@app.post("/saveConv/")
async def saveConv(saveConv: str = Form(...)):
    global selectedSaveConv
    selectedSaveConv = saveConv
    logger.info(f"save_conversation: {selectedSaveConv}")

@app.post("/selectLanguage/{lecture_id}")
async def selectLanguage(lecture_id: str, languageName: str = Form(...)):
    # Ensure that lecture state exists for the given lecture_id
    if lecture_id not in lecture_states:
        lecture_states[lecture_id] = { "state_machine": LectureStateMachine(lecture_id), "selectedLanguageName": languageName, "sessions": {}}
    else:
        lecture_states[lecture_id]["selectedLanguageName"] = languageName

    # logger.info(f"Selected Language for {lecture_id}: {languageName}")
    return {"message": "Language updated successfully", "selectedLanguageName": languageName}

@app.post("/changeLanguage/{lecture_id}")
async def change_language(lecture_id: str, languageName: str = Form(...)):
    if lecture_id not in lecture_states:
        return {"message": "Lecture not started yet."}
    # Update the language for all active sessions
    for session_id in lecture_states[lecture_id]["sessions"]:
        lecture_states[lecture_id]["sessions"][session_id]["selectedLanguageName"] = languageName

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

async def handle_user_message(
        websocket: WebSocket, 
        lecture_state: Dict, 
        message, 
        robot_id,
        db=Depends(get_db)):
    """Processes user message and sends response if WebSocket is open."""

    global stored_users  # Declare the global variable to modify it

    # Fetch the selected user
    selected_user = get_selected_student(robot_id)
    logger.info(f"selected_user: {selected_user}")

    # Check if the robot_id exists in stored_users
    existing_user = next((user for user in stored_users if user[0] == robot_id), None)

    logger.info("Start handle_user_message function successfully!")

    # user_message = json.loads(message).get("text")
    # user_name = json.loads(message).get("username")
    # total_count = json.loads(message).get("totalCount")
    # handsup_count = json.loads(message).get("handsUpCount")

    # user_data = db.users.find_one({"name": user_name})
    
    # overview = user_data.get("overview") if user_data else None

    logger.info("Start handle_user_message function successfully!")

    if message == "robot_text":
        result = "If you have any question, feel free to ask!"
        
        selectedLanguageName = lecture_state.get("selectedLanguageName", "English") or "English"
        session_id = str(websocket.client)
        #tts
        audio_stream = generate_audio_stream(result, selectedLanguageName)
        audio_length = get_audio_length(audio_stream)
        audio_stream.seek(0)
        audio_base64 = base64.b64encode(audio_stream.read()).decode("utf-8")

        # await websocket.send_text({"text": result, "audio": audio_base64, "type": "model"})
        
        data = {
            "robot_id":robot_id,
            "text": result,
            "audio": audio_base64,
            "type": "model"
        }

        logger.info(f"Data to be sent to audio client: robot_id: {robot_id}, text: {result}, type: model")

        # Convert the dictionary to a JSON string
        json_data = json.dumps(data)

        # send to audio client
        await websocket.send_text(json_data)
        return audio_length

    if lecture_state:
        lecture_state["last_message_time"] = time.time()

    selectedLanguageName = lecture_state.get("selectedLanguageName", "English") or "English"
    session_id = str(websocket.client)

    history = chat_histories.get(session_id, [])
    history = [sanitize_text(h) for h in history]

    retrieved_docs = faiss_text_db.similarity_search(message, k=5)
    retrieved_texts = "\n".join(sanitize_text(doc.page_content) for doc in retrieved_docs) if retrieved_docs else "No relevant context found."

    logger.info("History for Formatted Prompt: %s", history)
    logger.info("User Query for Formatted Prompt: %s", message)
    greeting_msg = "Hello"
    hour = local_time_set[robot_id]

    # Create a prompt for OpenAI to generate a greeting
    prompt = f"Generate a greeting message based on the current hour {hour}. Instead of 'Hello', make exact greeting based on current hour. And don't tell me about the exact time. Make this reply with one sentence. For example, 'Good morning', 'Good afternoon', 'Good evening' or 'Good night'"

    # Call OpenAI API to generate the greeting Chendra
    greeting_msg =  await predict(prompt)

    formatted_prompt = build_chat_prompt(
        custom_template=custom_prompt_template,
        history=history,                 # pass the list, not the joined string
        query=message,
        context=retrieved_texts,
        language="",
        username="",
        total_count="25",
        hands_up_count="5",
        overview="",
        greeting_msg = greeting_msg,
        )
    if formatted_prompt:
        logger.info("Prompty for LLM is ready successfully!")
    else:
        logger.error("Prompty for LLM is not ready.")
    #llm
  
    result = await predict(formatted_prompt)

    if result:
        logger.info("Result is generated successfully!")
    else:
        logger.error("Result is not generated.")
    #tts
    audio_stream = generate_audio_stream(result, selectedLanguageName)
    audio_length = get_audio_length(audio_stream)
    audio_stream.seek(0)
    audio_base64 = base64.b64encode(audio_stream.read()).decode("utf-8")

    # await websocket.send_text({"text": result, "audio": audio_base64, "type": "model"})
    
    data = {
        "robot_id":robot_id,
        "text": result,
        "audio": audio_base64,
        "type": "model"
    }

    logger.info(f"Data to be sent to audio client: robot_id: {robot_id}, text: {result}, type: model")

    # Convert the dictionary to a JSON string
    json_data = json.dumps(data)

    # send to audio client
    await websocket.send_text(json_data)

    quesAndAnswer = {
        "question": message,
        "answer": result,
        "model": selectedModelName,
        "prompt": custom_prompt_template
    }

    try:
        qna_document = QnA(**quesAndAnswer)
        logger.info(f"save_conversation: {selectedSaveConv}")
        if selectedSaveConv == "save":
            db.qna.insert_one(qna_document.dict(exclude_unset=True))
            logger.info("QnA successfully inserted into the database successfully!")
        if selectedSaveConv == "unsave":
            logger.info("QnA is not inserted into the database!")
    except Exception as e:
        logger.error(f"Error inserting QnA into database: {e}")

    if lecture_state:
        lecture_state["last_message_time"] = time.time()

    # history.append(f"User: {message}")
    # history.append(f"Assistant: {result}")

    # if len(history) > MAX_HISTORY_LENGTH * 2:
    #     history = history[-MAX_HISTORY_LENGTH * 2:]
    
    # chat_histories[session_id] = history
    
    is_new_user = existing_user is None
    is_changed_user = not is_new_user and selected_user != existing_user[1]

    if is_new_user:
        stored_users.append((robot_id, selected_user))
        logger.info("---------------------Initial selected_user stored.")
    elif is_changed_user:
        stored_users = [(robot_id, selected_user) if user[0] == robot_id else user for user in stored_users]
        logger.info("---------------------Changed")
    else:
        logger.info("---------------------Same")

    # Prepare or update chat history
    if is_new_user or is_changed_user:
        history = []
    else:
        history = chat_histories.get(session_id, [])

    history.append(f"User: {message}")
    history.append(f"Assistant: {result}")

    # Trim history if needed
    if len(history) > MAX_HISTORY_LENGTH * 2:
        history = history[-MAX_HISTORY_LENGTH * 2:]

    chat_histories[session_id] = history

    return audio_length

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

    # Insert into MongoDB
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

method = "Whisper"
@app.post("/sttMethod/")
async def sttMethod(sttMethod: str = Form(...)):    
    global method
    method = sttMethod

app.mount("/static", StaticFiles(directory="static"), name="static") 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)