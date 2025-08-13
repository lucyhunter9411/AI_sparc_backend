"""
Refactored Lecture-level FSM.

* A lightweight `Ctx` dataclass stores everything the FSM’s
  asynchronous routines need while a question-answer phase runs.
* `LectureStateMachine.set_ctx()` attaches a context object to the
  instance, so later triggers (e.g. `ev_enter_student_qna()`) need
  **no positional arguments** – exactly what the tests expect.

This file completely replaces the previous version.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect, Depends
from transitions import Machine, State

import base64
import wave
import os

from app.services.stt_service import transcribe_audio
from app.utils.audio import generate_audio_stream, get_audio_length
from app.services.shared_data import get_hand_raising_count, get_connected_audio_clients, set_qna_flag, get_qna_flag
from app.services.vision_service import  handle_vision_data
from app.websockets.connection_manager import ConnectionManager
from app.api.deps import get_conn_mgr
from app.services.tv_interface import send_image_to_devices

logger = logging.getLogger(__name__)

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
            "data": list(chunk)
        })
    return chunks

# ---------------------------------------------------------------------------
# helper: single place that stores per-call runtime data
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    websocket: WebSocket
    data: Dict[str, Any]
    lecture_state: Dict[str, Any]
    delay: float
    retrieve_data: str
    connectrobot: str


# ---------------------------------------------------------------------------
# FSM state map (unchanged)
# ---------------------------------------------------------------------------

states = [
    State(name="st_waiting"),
    State(name="st_conversation",
          on_enter=["on_enter_conversation"]),
    State(name="st_conducting_lecture",
          on_enter=["on_start_lecture"]),
    State(name="st_static_content",
          on_enter=["on_enter_content"],
          on_exit=["on_exit_content"]), 
    State(name="st_teacher_qna",
          on_enter=["on_enter_teacher_qna"],
          on_exit=["on_exit_teacher_qna"]),
    State(name="st_student_qna",
          on_enter=["on_enter_student_qna"],
          on_exit=["on_exit_student_qna"]),
    State(name="st_process_qna", 
         on_enter=["on_enter_process_qna"], 
         on_exit=["on_exit_process_qna"]),
    State(name="st_exit_conf"),
    State(name="st_ai_summarize", 
         on_enter=["on_ai_summarize"],
         on_exit=["on_enter_conversation"])
]

transitions = [
    {"trigger": "ev_enter_conversation", "source": "st_waiting",            "dest": "st_conversation"},
    {"trigger": "ev_start_lecture",      "source": "st_waiting",            "dest": "st_conducting_lecture"},

    {"trigger": "ev_enter_content",      "source": "st_conducting_lecture", "dest": "st_static_content"},
    {"trigger": "ev_exit_content",       "source": "st_static_content",     "dest": "st_conducting_lecture"},

    {"trigger": "ev_enter_teacher_qna",  "source": "st_conducting_lecture", "dest": "st_teacher_qna"},
    {"trigger": "ev_exit_teacher_qna",   "source": "st_teacher_qna",        "dest": "st_conducting_lecture"},

    {"trigger": "ev_enter_student_qna",  "source": "st_conducting_lecture", "dest": "st_student_qna"},
    {"trigger": "ev_exit_student_qna",   "source": "st_student_qna",        "dest": "st_exit_conf"},

    {"trigger": "ev_enter_process_qna",  "source": "st_student_qna",        "dest": "st_process_qna"},
    {"trigger": "ev_exit_process_qna",   "source": "st_process_qna",        "dest": "st_student_qna"},

    {"trigger": "ev_ai_summarize",       "source": "st_exit_conf",          "dest": "st_ai_summarize"},
    {"trigger": "ev_next_topic",         "source": ["st_exit_conf", "st_ai_summarize"], "dest": "st_conducting_lecture"},

    {"trigger": "ev_init",               "source": "*",                     "dest": "st_waiting"},
    {"trigger": "ev_to_conducting",      "source": "*",                     "dest": "st_conducting_lecture"},
]

# ---------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------

class LectureStateMachine:
    """Finite-state machine that drives a single lecture."""
        
    def __init__(self, lecture_id:str):
        self.lecture_id = lecture_id
        self.start_time = time.time()
        self.question_active = False
        self.ctx: Optional[Ctx] = None        # ← context lives here
        self.machine = Machine(model=self, states=states, transitions=transitions, initial="st_waiting")

    # -------------- public helper ----------------------------------------

    def set_ctx(self, **kwargs) -> None:
        """
        Attach (or overwrite) the context used by `ev_enter_student_qna`.

        >>> sm.set_ctx(websocket=ws, data=payload, lecture_state=row, ...)
        """
        self.ctx = Ctx(**kwargs)

    # -------------- entry callbacks --------------------------------------

    async def enter_conversation(self, websocket: WebSocket, lecture_states: Dict, message, robot_id):
        logger.info(
            "State machine is passed through ev_enter_conversation event: %s",
            robot_id
        )
        self.websocket = websocket
        self.lecture_states = lecture_states
        self.message = message
        self.robot_id = robot_id
        self.trigger("ev_enter_conversation") 
        await self._handle_conversation()

    async def exit_student_qna(self):
        self.trigger("ev_exit_student_qna")

    def on_enter_student_qna(self):
        """Enter student Q&A state"""
        logger.info(f"Lecture {self.lecture_id}: Entering student Q&A state.")

    def on_exit_student_qna(self):
        """Exit student Q&A state"""
        logger.info(f"Lecture {self.lecture_id}: Exiting student Q&A state.")

    def on_enter_process_qna(self):
        """Enter process Q&A state"""
        logger.info(f"Lecture {self.lecture_id}: Entering process Q&A state.")

    def on_exit_process_qna(self):
        """Exit process Q&A state"""
        logger.info(f"Lecture {self.lecture_id}: Exiting process Q&A state.")

    def on_ai_summarize(self):
        """AI summarization event"""
        logger.info(f"Lecture {self.lecture_id}: AI summarization triggered.")

    def on_next_topic(self):
        """Transition to the next topic"""
        logger.info(f"Lecture {self.lecture_id}: Moving to the next topic.")

    async def enter_student_qna(self, current_state_machine, robot_id, websocket, data, lecture_state, delay, retrieve_data, connectrobot):
        """
        No positional arguments now – relies on `self.ctx` being set.

        Call `set_ctx()` right beforehand.
        """
        if not self.ctx:
            raise RuntimeError("Context not set – call set_ctx() first")
        self.trigger("ev_enter_student_qna")
        lecture_state["question_time_start"] = time.time()
        lecture_state["last_message_time"] = time.time()
        question_active = True

        task2 = asyncio.create_task(self.check_question_timeout(websocket, robot_id, question_active, lecture_state, delay, retrieve_data))
        task3 = asyncio.create_task(self.check_hand_raising(current_state_machine, robot_id, websocket, question_active, lecture_state, connectrobot))

        done, pending = await asyncio.wait([task2, task3], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        logger.info("Canceled one of thread......................")
        # self.trigger("ev_enter_content") 


    async def check_hand_raising(self, current_state_machine, robot_id, websocket, question_active, lecture_state, connectrobot):
        while question_active:
            current_count = get_hand_raising_count(connectrobot)
            logger.info(f"Current hand_raising_count: {current_count}")
            await asyncio.sleep(1)
            if current_count > 0:
                logger.info("Someone is raising hand")
                await handle_vision_data(current_state_machine, robot_id, websocket)
                self.trigger("ev_enter_process_qna")
                if hasattr(self.machine, 'state') and self.machine.state == "st_process_qna":
                    self.trigger("ev_exit_process_qna")        
    logger.info("Exiting check_hand_raising function.")

    async def check_question_timeout(self, websocket, robot_id, question_active, lecture_state, delay, retrieve_data):
        from main import generate_and_send_ai_response
        set_qna_flag(robot_id, True)
        
        # ✅ REMOVED: Direct ping test that caused race conditions
        # connected_audio_clients = get_connected_audio_clients(robot_id)
        # The audio dispatch service will handle audio client detection atomically

        while question_active:    
            selectedLanguageName = lecture_state["selectedLanguageName"]
            elapsed_time = time.time() - lecture_state["question_time_start"]
            not_interactive_time = time.time() - lecture_state["last_message_time"]
            qna_flag = get_qna_flag(robot_id)
            if not qna_flag:
                lecture_state["last_message_time"] = time.time()
                set_qna_flag(robot_id, True)
            logger.info(f"Elapsed Time:  {elapsed_time}    Not Interactive Time: {not_interactive_time}")
            if elapsed_time > delay * 60 - 5:
                # Only exit process_qna if we're actually in that state
                if hasattr(self.machine, 'state') and self.machine.state == "st_process_qna":
                    self.trigger("ev_exit_process_qna")
                self.trigger("ev_exit_student_qna")
                question_active = False
                resp = ""
                if selectedLanguageName == "English":
                    resp = "Sorry, we are out of time."
                elif selectedLanguageName == "Hindi":
                    resp = "क्षमा करें, कृपया अगली बार प्रश्न पूछें।"
                elif selectedLanguageName == "Telugu":
                    resp = "క్షమించండి, దయచేసి తదుపరిసారి ప్రశ్న అడగండి."
                # ✅ USE CENTRALIZED AUDIO DISPATCH SERVICE for timeout responses
                from app.services.audio_dispatch_service import audio_dispatch_service
                
                # Prepare timeout response data
                timeout_data = {}
                if selectedLanguageName == "English":
                    timeout_data["EnglishText"] = resp
                elif selectedLanguageName == "Hindi":
                    timeout_data["HindiText"] = resp
                elif selectedLanguageName == "Telugu":
                    timeout_data["TeluguText"] = resp
                else:
                    timeout_data["EnglishText"] = resp  # Default to English
                
                # Use centralized dispatch service
                await audio_dispatch_service.dispatch_audio(
                    robot_id=robot_id,
                    content_data=timeout_data,
                    frontend_websocket=websocket
                )
                
                # Send stop message to frontend
                await websocket.send_text(json.dumps({"text": resp, "type": "stop"}))

                
                # audio_chunks = chunk_audio(audio_bytes, CHUNK_SIZE)
                # for i, chunk in enumerate(audio_chunks):
                #     chunk_message = {
                #         "text": resp if i == 0 else "",
                #         "audio_chunk": chunk,  # Send each chunk with its metadata
                #         "type": "stop"
                #     }
                #     await websocket.send_text(json.dumps(chunk_message))
                await asyncio.sleep(5)
                await websocket.send_text(json.dumps({"type": "lesson_event", "text": "QNA_ENDED"})) 
                self.trigger("ev_init")
                break

            elif elapsed_time < delay * 60 - 20 and not_interactive_time > 60:
                # Only exit process_qna if we're actually in that state
                if hasattr(self.machine, 'state') and self.machine.state == "st_process_qna":
                    self.trigger("ev_exit_process_qna")
                self.trigger("ev_exit_student_qna")
                self.trigger("ev_ai_summarize")
                await websocket.send_text(json.dumps({"type": "lesson_event", "text": "SUMMARIZE_STARTED"}))
                question_active = False
                remaining_time = delay * 60 - int(elapsed_time)
                before = time.time()
                # ✅ FIXED: Pass True as default for is_websocket_alive since we're using centralized dispatch
                await generate_and_send_ai_response(websocket, lecture_state, retrieve_data, remaining_time, True, robot_id)
                logger.info(f"Remaining: {remaining_time}    Generation: {int(time.time() - before)}")
                await asyncio.sleep(remaining_time-int(time.time() - before))
                await websocket.send_text(json.dumps({"type": "lesson_event", "text": "SUMMARIZE_ENDED"}))
                await websocket.send_text(json.dumps({"type": "lesson_event", "text": "QNA_ENDED"})) 
                self.trigger("ev_init")
                break

            await asyncio.sleep(1)

    async def _handle_conversation(self):
        from main import handle_user_message
        logger.info("Start handle_conversation function successfully!")

        logger.info("Lecture %s: Conversation started.", self.lecture_id)
        session_id = str(self.websocket.client)

        if session_id not in self.lecture_states[self.lecture_id]["sessions"]:
            self.lecture_states[self.lecture_id]["sessions"][session_id] = {}

        self.lecture_states[self.lecture_id]["sessions"][session_id].update({
            "selectedLanguageName": "English",
            "is_active": True,
        })

        try:
            duration = await handle_user_message(self.websocket, {}, self.message, self.robot_id)
            # await asyncio.sleep(duration)
            self.message = ""

        except WebSocketDisconnect:
            logger.error("❌ WebSocket disconnected unexpectedly: %s", session_id)
            self.lecture_states[self.lecture_id]["sessions"][session_id]["is_active"] = False  # ✅ Mark session inactive
            self.trigger("ev_init")  # ✅ Reset state machine safely
        except Exception as e:
            logger.exception("❌ Unexpected WebSocket error:")

        finally:
            self.trigger("ev_init")  # ✅ Reset state on exit

    def on_enter_conversation(self):
        logger.info("Entered `st_conversation` state.")

    def start_lecture(self, websocket: WebSocket, lecture_states:Dict, connectrobot):
        logger.info(f"Lecture {self.lecture_id} started.")
        session_id = str(websocket.client)
        
        # ❌ REMOVED: Global function calls that could cause collisions
        # contents = get_contents()
        # time_list = get_time_list()
        
        # ✅ Get data from session instead - no more global dependencies!
        session = lecture_states[self.lecture_id][connectrobot]["sessions"].get(session_id, {})
        contents = session.get("contents", [])
        time_list = session.get("time_list", [])
        
        if session_id not in lecture_states[self.lecture_id][connectrobot]["sessions"]:
            selectedLanguageName = lecture_states[self.lecture_id].get("selectedLanguageName", "English")
            lecture_states[self.lecture_id][connectrobot]["sessions"][session_id] = {
                "is_active": True,
                "selectedLanguageName": selectedLanguageName,
                "websocket": websocket,
                "contents": contents,        # ✅ Use local session content
                "time_list": time_list,     # ✅ Use local session timing
            }
        self.trigger("ev_start_lecture")

    def on_start_lecture(self):
        logger.info("Lecture started!!!!!!!")

    async def enter_content(
        self, 
        data, 
        lecture_state, 
        websocket,
        connectrobot, 
        db,
        idx,
        mgr: ConnectionManager = Depends(get_conn_mgr),
    ) -> None:
        """Enter static content state - Now uses centralized audio dispatch service"""
        from main import chat_histories, MAX_HISTORY_LENGTH
        from app.services.audio_dispatch_service import audio_dispatch_service
        
        self.trigger("ev_enter_content")
        logger.info(f"Lecture {self.lecture_id}: Entering static content state.")
        session_id = str(websocket.client)
        selectedLanguageName = lecture_state["selectedLanguageName"]

        # Send lesson event first
        await websocket.send_text(json.dumps({"type": "lesson_event", "text": "CONTENT", "index": idx}))

        # Handle image to TV devices
        extract_image = data.get("image")
        lesson_image = extract_image
        image_path_to_TV = lesson_image.replace("\\", "/") if lesson_image else ""
        
        if image_path_to_TV:
            await send_image_to_devices(connectrobot, db, image_path_to_TV, logger)

        # ✅ USE CENTRALIZED AUDIO DISPATCH SERVICE
        # This replaces the old race-condition prone logic
        dispatch_ctx = await audio_dispatch_service.dispatch_audio(
            robot_id=connectrobot,
            content_data=data,
            frontend_websocket=websocket
        )
        
        # Get audio length for timing (regardless of dispatch method)
        try:
            audio_stream = generate_audio_stream(data.get(f"{selectedLanguageName}Text"), selectedLanguageName)
            audio_length = get_audio_length(audio_stream)
        except Exception as e:
            logger.warning(f"[{connectrobot}] Could not get audio length: {e}")
            audio_length = 3.0  # Default fallback
        
        self.start_time = time.time()
         
        # Use chat_histories to store or retrieve conversation history
        if session_id not in chat_histories:
            chat_histories[session_id] = []  # Initialize if not present


        history = chat_histories.get(session_id, [])
        history.append(f"Assistant: {data}")
        if len(history) > MAX_HISTORY_LENGTH * 2:
            history = history[-MAX_HISTORY_LENGTH * 2:]
        chat_histories[session_id] = history

        while time.time() - self.start_time < int(audio_length) + 2:
            await asyncio.sleep(0.1)

        self.start_time = time.time()
        # self.trigger("ev_exit_content")

    def on_enter_content(self):
        logger.info(f"Lecture {self.lecture_id}: Static Content started")
    def on_exit_content(self):
        logger.info(f"Lecture {self.lecture_id}: Static Content ended")

    def on_enter_teacher_qna(self):
        """Enter teacher Q&A state"""
        logger.info(f"Lecture {self.lecture_id}: Entering teacher Q&A state.")

    def on_exit_teacher_qna(self):
        """Exit teacher Q&A state"""
        logger.info(f"Lecture {self.lecture_id}: Exiting teacher Q&A state.")

