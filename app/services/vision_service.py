import asyncio  # For asynchronous programming
import base64  # For encoding audio data to base64
import json  # For converting data to JSON format
from app.services.shared_data import get_hand_raising_count, get_name_array, set_hand_raising_count, set_name_array, set_selected_student
from app.utils.audio import generate_audio_stream, get_audio_length
from app.services.llm_service import generate_openai_response
from datetime import datetime
from pydantic import BaseModel
import logging  # For logging information

# Configure the logger
logging.basicConfig(level=logging.INFO)  # Set the logging level
logger = logging.getLogger(__name__)  # Create a logger object
        
class VisionData(BaseModel):
    handup_result: list
    face_recognition_result: list
    robot_id: str
    image_name: str  # The name of the image file
    image: str  # The image data in bytes
    detect_user: list

async def handle_vision_data(current_state_machine, robot_id, websocket):
    logger.info("handle_vision_data function is called correctly.")
    while True:
        # Use the latest robot_id
        current_count = get_hand_raising_count(robot_id)
        current_user = get_name_array(robot_id)
        # if current_count > 0:
            # logger.info(f"Current connectrobot: {robot_id}")
            # logger.info(f"Current hand_raising_count: {current_count}")
            # logger.info(f"Current hand_raising_count: {current_user}")
            # logger.info(f"Current current_state_machine: {current_state_machine}")
            # logger.info(f"Current websocket: {websocket}")

        # Always get the current state based on the latest lecture_id
        if current_state_machine:
            current_state = current_state_machine.state  # Get the current state name
            # logger.info(f"current_state {current_state}")

            if current_count > 0 and current_state == "st_waiting":
                local_time_vision = datetime.now().isoformat()
                dt_vision = datetime.fromisoformat(local_time_vision)
                formatted_time_vision = dt_vision.strftime("%H")
                hour = int(formatted_time_vision)  # Convert to integer

                # Get the student's name
                # student_name = current_user[0] if current_user else "Student"  # Default to "Student" if no name
                student_name = "Student"  # Default to "Student" if no name
                if isinstance(current_user, list):
                    for name in current_user:
                        if name != "Unknown":  # Check if the name is not "Unknown"
                            student_name = name  # Use the first valid name found
                            break  # Exit the loop once a valid name is found
                    set_selected_student(robot_id, student_name)
                else:
                    logger.error(f"Expected current_user to be a list, but got: {type(current_user)} with value {current_user}")

                # Create a prompt for OpenAI to generate a greeting
                prompt = f"Generate a greeting message for a student named {student_name} based on the current hour {hour}. The message should invite the student to ask questions. Instead of 'Hello', make exact greeting based on current hour. And don't tell me about the exact time. Make this reply with 2 sentences. One sentence is greeting like 'Good morning, Student' and other sentence is to ask them to make question."

                # Call OpenAI API to generate the greeting
                result = generate_openai_response(prompt)
                logger.info(f"Current result: {result}")
                # result = "If you have any questions, feel free to ask!"
                selectedLanguageName = "English"
                # TTS
                audio_stream = generate_audio_stream(result, selectedLanguageName)
                audio_length = get_audio_length(audio_stream)
                audio_stream.seek(0)
                audio_base64 = base64.b64encode(audio_stream.read()).decode("utf-8")
                data = {
                    "robot_id": robot_id,
                    "text": result,
                    "audio": audio_base64,
                    "type": "model"
                }
                logger.info(f"Data to be sent to audio client: robot_id: {robot_id}, text: {result}, type: model")
                logger.info(f"audio_length: {audio_length}")

                # Convert the dictionary to a JSON string
                json_data = json.dumps(data)

                # Send to audio client
                await websocket.send_text(json_data)
                await asyncio.sleep(audio_length + 5)  # Adjust the sleep time as needed

        await asyncio.sleep(1)  # Adjust the sleep time as needed

async def get_data(vision_data: VisionData): # rename visionUpdate

    logger.info("get_data function is runnning.")

    # Process the received data
    handup_result = vision_data.handup_result
    face_recognition_result = vision_data.face_recognition_result
    robot_id = vision_data.robot_id
    image_name = vision_data.image_name
    image = vision_data.image
    detect_user = vision_data.detect_user

    names_array = [entry['name'] for entry in detect_user]
    set_name_array(robot_id, names_array)

    hand_raising_count = sum(1 for item in handup_result if item.get('label') == 'hand-raising')
    set_hand_raising_count(robot_id, hand_raising_count)