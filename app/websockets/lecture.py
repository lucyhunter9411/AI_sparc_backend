# lecture.py

import json
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Form, Depends
from app.state.lecture_state_machine import LectureStateMachine
from typing import Dict, List
import logging
from app.api.deps import get_db
from app.services.shared_data import set_contents, set_time_list, set_session_id_set, get_session_id_set, set_lecture_states, get_lecture_states, get_language_selected, set_topic_title, get_topic_title
import uuid
import asyncio
import base64
from app.utils.audio import generate_audio_stream, get_audio_length
import os
from bson import ObjectId

router = APIRouter()

# Initialize logger
logger = logging.getLogger(__name__)

# Additional global variables
lecture_to_audio = {}
data_to_audio = {}
current_state_machine = None
robot_id_before = None

from typing import Dict, List

contents = []
time_list = []
topics = []

CHUNK_SIZE = os.getenv("CHUNK_SIZE")

# Convert CHUNK_SIZE to an integer, with a default value if not set or invalid
try:
    CHUNK_SIZE = int(CHUNK_SIZE)
except (TypeError, ValueError):
    # Set a default value if CHUNK_SIZE is not set or is not a valid integer
    CHUNK_SIZE = 2097152  # Example default value, adjust as needed - 2MB

def chunk_audio(audio_data, chunk_size):
    """Split audio data into chunks with sequence numbers and total count."""
    total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = audio_data[start:end]
        chunks.append({
            "sequence_number": i,
            "total_chunks": total_chunks,
            # "data": list(chunk)
            "data": base64.b64encode(chunk).decode('utf-8')  # Use base64 instead of list
        })
    return chunks

@router.post("/startLecture/{lecture_id}/{topic_id}")
async def start_lecture(lecture_id: str, topic_id: str, connectrobot: str = Form(...), db=Depends(get_db)):
    global topics, contents, time_list
    topics = list(db.topics.find({"lecture_id": lecture_id}).sort("_id", 1))

    start_index = next((i for i, topic in enumerate(topics) if str(topic["_id"]) == str(topic_id)), None)
    # topics_lecture = topics[start_index:]
    topics_lecture = [topics[start_index]] if start_index is not None else []
    contents = []
    time_list = []
    for topic in topics_lecture:
        for content in topic.get("content", []):
            contents.append(content)
            time_list.append(content["time"])
        contents.append({"text": "question", "time": topic["qna_time"]})
        time_list.append(topic["qna_time"])

    set_contents(contents)
    set_time_list(time_list)
    lecture_states = get_lecture_states()

    if lecture_id not in lecture_states:
        lecture_states[lecture_id] = {}

    if connectrobot not in lecture_states[lecture_id]:
        lecture_states[lecture_id][connectrobot] = {
            "state_machine": LectureStateMachine(lecture_id),
            "sessions": {}
        }

    session_id = str(uuid.uuid4())
    set_session_id_set(connectrobot, session_id)
    language_selected = get_language_selected()
    
    # Use get method to safely access the selected language
    selected_language = language_selected.get(lecture_id, {}).get(connectrobot, {}).get("selectedLanguageName", "English")
    
    lecture_states[lecture_id][connectrobot]["sessions"][session_id] = {
        "is_active": True,
        "selectedLanguageName": selected_language,
        "contents": contents,
        "time_list": time_list,
        "connectrobot": connectrobot,
    }
    set_lecture_states(lecture_states)
    
    # Get lecture title from the database
    lecture = db.lectures.find_one({"_id": ObjectId(lecture_id)})
    lecture_title = lecture.get("title", "Unknown Lecture") if lecture else "Unknown Lecture"
    
    # Get topic title and QnA time from the current topic
    current_topic = next((topic for topic in topics if str(topic["_id"]) == str(topic_id)), None)
    topic_title = current_topic.get("title", "Unknown Topic") if current_topic else "Unknown Topic"
    set_topic_title(connectrobot, topic_title)
    topic_qna_time = current_topic.get("qna_time", 0) if current_topic else 0
    
    # Create return data
    return_data = {
        "status": "Lecture started", 
        "lecture_id": lecture_id, 
        "session_id": session_id,
        "lecture_title": lecture_title,
        "topic_title": topic_title,
        "topic_qna_time": topic_qna_time
    }
    
    # Log the return values
    logger.info(f"✅ Lecture started successfully - Lecture: '{lecture_title}' (ID: {lecture_id}), Topic: '{topic_title}' (ID: {topic_id}), QnA Time: {topic_qna_time}s, Session: {session_id}, Robot: {connectrobot}")
    
    return return_data

@router.websocket("/ws/{robot_id}/lesson_audio")
async def websocket_lesson_audio(websocket: WebSocket, robot_id: str):
    global data_to_audio, lecture_to_audio  # use globals from this module
    await websocket.accept()
    try:
        while True:
            if len(data_to_audio):
                if robot_id in data_to_audio:  # Check if the robot_id exists in data_to_audio
                    if data_to_audio[robot_id].get("data"):
                        selectedLanguageName = lecture_to_audio[robot_id]["selectedLanguageName"]
                        audio_stream = generate_audio_stream(
                            data_to_audio[robot_id]["data"].get(f"{selectedLanguageName}Text"), selectedLanguageName
                        )

                        audio_length = get_audio_length(audio_stream)
                        audio_stream.seek(0)
                        # audio_base64 = base64.b64encode(audio_stream.read()).decode("utf-8")
                        audio_bytes = audio_stream.read()  # Read the audio as bytes
                        
                        audio_chunks = chunk_audio(audio_bytes, CHUNK_SIZE)
                        logger.info(f"[{robot_id}] Created {len(audio_chunks)} audio chunks for audio client")
                    
                        # You can process the data or send a response back
                        for i, chunk in enumerate(audio_chunks):
                            message_data = {
                                "robot_id": robot_id,
                                "text": data_to_audio[robot_id]["data"].get(f"{selectedLanguageName}Text") if i == 0 else "",
                                "audio_chunk": chunk,
                                "type": "to_audio_client",
                            }
                            
                            # Convert to JSON string to check size
                            json_message = json.dumps(message_data)
                            message_size = len(json_message.encode('utf-8'))  # Size in bytes
                            
                            logger.info(f"[{robot_id}] Sending WebSocket message chunk {i+1}/{len(audio_chunks)}, size: {message_size:,} bytes ({message_size/1024:.2f} KB)")
                            
                            await websocket.send_text(json_message)
                        data_to_audio[robot_id]["data"] = None
            await asyncio.sleep(0.1)  # Reduced delay to catch data more quickly
    except WebSocketDisconnect:
        logger.error("Client disconnected from testdata WebSocket")

# WebSocket endpoint for lecture
@router.websocket("/ws/{lecture_id}/{connectrobot}")
async def lecture_websocket_endpoint(websocket: WebSocket, lecture_id: str, connectrobot: str, db=Depends(get_db)):
    global current_state_machine, robot_id_before
    await websocket.accept()
    lecture_states = get_lecture_states()
    session_id = get_session_id_set(connectrobot)

    # Check if the lecture_id exists in lecture_states
    if lecture_id not in lecture_states or connectrobot not in lecture_states[lecture_id]:
        # Initialize a new entry for the lecture_id and connectrobot
        lecture_states[lecture_id][connectrobot] = {
            "state_machine": LectureStateMachine(lecture_id),
            "sessions": {}
        }

    # Check if the session_id already exists for this lecture_id and connectrobot
    if session_id not in lecture_states[lecture_id][connectrobot]["sessions"]:
        # Initialize a new session for this lecture_id and connectrobot
        lecture_states[lecture_id][connectrobot]["sessions"][session_id] = {
            "is_active": True,
            "selectedLanguageName": "English",
            # You can add other session-specific data here
        }

    state_machine = lecture_states[lecture_id][connectrobot]["state_machine"]
    current_state_machine = state_machine
    state_machine.ev_init() 
    state_machine.start_lecture(websocket, lecture_states, connectrobot)
    lecture_to_audio[connectrobot] = lecture_states[lecture_id][connectrobot]["sessions"][session_id]
    retrieve_data = "\n"

    await websocket.send_text(json.dumps({"type": "lesson_event", "text": "LECTURE_STARTED", "topic_title": get_topic_title(connectrobot)}))  

    try: 
        for idx, delay in enumerate(lecture_to_audio[connectrobot]["time_list"]):   
            state_machine.ev_to_conducting()
            data_frontend = lecture_to_audio[connectrobot]["contents"][idx]
            # logger.info("data_frontend: %s", data_frontend)
            if connectrobot not in data_to_audio:
                data_to_audio[connectrobot] = {}

            retrieve_data += data_frontend.get("text", "") + "\n"

            if data_frontend.get("text") == "question":
                await websocket.send_text(json.dumps({"type": "lesson_event", "text": "QNA_STARTED", "qna_time": delay})) 
                await websocket.send_text(json.dumps({"text": data_frontend.get("text")}))  
                state_machine.set_ctx(
                    websocket=websocket,
                    data=data_frontend,
                    lecture_state=lecture_to_audio[connectrobot],
                    delay=delay,
                    retrieve_data=retrieve_data,
                    connectrobot=connectrobot,
                )
                await state_machine.enter_student_qna(current_state_machine, connectrobot, websocket, data_frontend, lecture_to_audio[connectrobot], delay, retrieve_data, connectrobot)
                state_machine.ev_to_conducting()
            else: 
                data_to_audio[connectrobot]["data"] = data_frontend
                await state_machine.enter_content(data_frontend, lecture_to_audio[connectrobot], websocket, connectrobot, db, idx)
        await websocket.send_text(json.dumps({"type": "lesson_event", "text": "LECTURE_ENDED"})) 
                
        if not lecture_to_audio[connectrobot]["is_active"]:
            state_machine.ev_init()
            await websocket.close()
            return

    except WebSocketDisconnect:
        logger.exception("❌ Client disconnected unexpectedly.")
        del lecture_states[lecture_id][connectrobot]["sessions"][session_id]
        state_machine.ev_init()
        await websocket.close()