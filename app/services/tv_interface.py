import os
import httpx
import requests
import websockets
import json
from fastapi import HTTPException
from dotenv import load_dotenv
from pathlib import Path
from app.utils.classrooms import get_classrooms

# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

TV_app_api = os.getenv("TV_app_api")

def sign_in() -> dict:
    """Sign in to TV app using environment credentials"""
    TV_app_login_email = os.getenv("TV_app_login_email")
    TV_app_login_password = os.getenv("TV_app_login_password")
    
    form_data = {
        "username": TV_app_login_email,
        "password": TV_app_login_password
    }
    
    response = requests.post(f"{TV_app_api}/auth/token", data=form_data)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Login failed")
    
    return response.json()

# Wrapper function to use dependency injection
async def fetch_classrooms(robot_id: str, db):
    return await get_classrooms(robot_id, db)

async def send_image_to_devices(robot_id, db, image_path_to_TV, log):
    try:
        # Use the wrapper function to fetch classrooms
        # classrooms = await fetch_classrooms(robot_id, db)
        # log.info(f"Classrooms for robot_id {robot_id}: {classrooms}")
        # device_list = classrooms.get('classrooms', [])
        # device_ids = [classroom['device_id'] for classroom in device_list if 'device_id' in classroom]
        # log.info(f"Device IDs for robot_id {robot_id}: {device_ids}")

        device_ids = [robot_id]
        log.info(f"Device IDs for robot_id {robot_id}: {device_ids}")

        TV_app_websocket = os.getenv("TV_app_websocket")

        # Get authentication token
        auth_response = sign_in()
        access_token = auth_response.get("access_token")

        async with httpx.AsyncClient(timeout=10.0) as client:
            for device_id in device_ids:
                try:
                    # Define the API endpoint
                    api_endpoint = f"{TV_app_api}/devices/url/{device_id}"
                    headers = { 
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    }
                    # # Send the API request (correcting the payload key to 'url')
                    # base_url = os.getenv("base_url")
                    # image_path_to_TV = base_url + closest_image_path.replace("\\", "/")
                    response = await client.put(api_endpoint, json={"url": image_path_to_TV}, headers=headers)
                    log.info(f"Successfully sent API request for device_id {device_id}: {response}")
                    
                    # Send WebSocket messages
                    await send_lesson_state_change(device_id, access_token, TV_app_websocket, False, log)
                    await send_lesson_state_change(device_id, access_token, TV_app_websocket, True, log)

                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    log.error(f"HTTP error for device_id {device_id}: {e.response.text}")
                except httpx.RequestError as e:
                    log.error(f"Network error for device_id {device_id}: {e}")
                except Exception as e:
                    log.error(f"General error for device_id {device_id}: {e}")
    except HTTPException as e:
        log.error(f"HTTP error retrieving classrooms for robot_id {robot_id}: {e.detail}")
    except Exception as e:
        log.error(f"Error retrieving classrooms for robot_id {robot_id}: {e}")

async def send_lesson_state_change(device_id, access_token, websocket_url, is_active, log):
    """Send a lesson state change message via WebSocket."""
    uri = f"{websocket_url}/ws?token={access_token}"
    log.info(f"Connecting to WebSocket: {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            message = {
                "type": "lesson_state_change",
                "deviceId": device_id, 
                "isActive": is_active
            }
            await websocket.send(json.dumps(message))
            response = await websocket.recv()
            log.info(f"Received: {response}")
    except Exception as e:
        log.error(f"WebSocket error for device_id {device_id}: {e}") 