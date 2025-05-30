# lecture.py

import json
from fastapi import WebSocket, WebSocketDisconnect
from app.state.lecture_state_machine import LectureStateMachine
from app.services.shared_data import get_lecture_states
from typing import Dict, List
import logging

from fastapi import APIRouter, Form, Depends
from app.api.deps import get_db
from app.services.shared_data import set_contents, set_time_list, set_session_id_set, get_session_id_set
import uuid

router = APIRouter()

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables
lecture_states: Dict[str, Dict[str, Dict[str, Dict]]] = {}
chat_histories: Dict[str, List[str]] = {}
MAX_HISTORY_LENGTH = 5
stored_users = []

# Additional global variables
lecture_state_test = {}
data_test = {}
current_state_machine = None
robot_id_before = None
connected_clients = {}
connected_audio_clients = {}

from typing import Dict, List

contents = []
time_list = []
topics = []

@router.post("/startLecture/{lecture_id}/{topic_id}")
async def start_lecture(lecture_id: str, topic_id: str, connectrobot: str = Form(...), db=Depends(get_db)):
    global topics, contents, time_list
    topics = list(db.topics.find({"lecture_id": lecture_id}).sort("_id", 1))

    start_index = next((i for i, topic in enumerate(topics) if str(topic["_id"]) == str(topic_id)), None)
    topics_lecture = topics[start_index:]
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

    if lecture_id not in lecture_states:
        lecture_states[lecture_id] = {}

    if connectrobot not in lecture_states[lecture_id]:
        lecture_states[lecture_id][connectrobot] = {
            "state_machine": LectureStateMachine(lecture_id),
            "sessions": {}
        }

    session_id = str(uuid.uuid4())
    set_session_id_set(connectrobot, session_id)
    lecture_states[lecture_id][connectrobot]["sessions"][session_id] = {
        "is_active": True,
        "selectedLanguageName": "English",
        "contents": contents,
        "time_list": time_list,
        "connectrobot": connectrobot,
    }

    return {"status": "Lecture started", "lecture_id": lecture_id, "session_id": session_id}

# WebSocket endpoint for lecture
async def lecture_websocket_endpoint(websocket: WebSocket, lecture_id: str, connectrobot: str):

    global current_state_machine, robot_id_before

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
    state_machine.ev_start_lecture(websocket, lecture_states, connectrobot)
    print("-----------------", session_id, "--------------------")
    print(lecture_states[lecture_id][connectrobot]["sessions"])
    # print(lecture_states[lecture_id][connectrobot]["sessions"][session_id])
    lecture_state_test[connectrobot] = lecture_states[lecture_id][connectrobot]["sessions"][session_id]
    retrieve_data = "\n"

    try: 
        for idx, delay in enumerate(lecture_state_test[connectrobot]["time_list"]):   
            state_machine.ev_to_conducting()
            data_frontend = lecture_state_test[connectrobot]["contents"][idx]
            logger.info("data_frontend: %s", data_frontend)
            # before using `data_test[connectrobot]["data"] = ...`, add this:
            if connectrobot not in data_test:
                data_test[connectrobot] = {}

            retrieve_data += data_frontend.get("text", "") + "\n"

            if data_frontend.get("text") == "question":
                await websocket.send_text(json.dumps({"text": data_frontend.get("text")}))  
                state_machine.set_ctx(
                websocket=websocket,
                data=data_frontend,
                lecture_state=lecture_state_test[connectrobot],
                delay=delay,
                retrieve_data=retrieve_data,
                connectrobot=connectrobot,
            )
                await state_machine.ev_enter_student_qna(current_state_machine, robot_id_before, websocket, data_frontend, lecture_state_test[connectrobot], delay, retrieve_data, connectrobot)                
                state_machine.ev_to_conducting()
            else:
                data_test[connectrobot]["data"] = data_frontend
                logger.info("data_test: %s", data_test)
                await state_machine.ev_enter_content(data_frontend, lecture_state_test[connectrobot], websocket, connected_clients, connected_audio_clients, connectrobot)

        if not lecture_state_test[connectrobot]["is_active"]:
            state_machine.ev_init()
            await websocket.close()
            return

    except WebSocketDisconnect:
        logger.exception("❌ Client disconnected unexpectedly.")
        del lecture_states[lecture_id][session_id]
        state_machine.ev_init()
        await websocket.close()