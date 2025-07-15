import os
import httpx
from fastapi import HTTPException
from app.utils.rooms import get_rooms  
from app.utils.TV_app import sign_in 

from dotenv import load_dotenv
from pathlib import Path
import websockets
import json

# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# Wrapper function to use dependency injection
async def fetch_rooms(robot_id: str, db):
    return await get_rooms(robot_id, db)

async def send_image_to_devices(robot_id, db, closest_image_path, log):
    try:
        # Use the wrapper function to fetch rooms
        rooms = await fetch_rooms(robot_id, db)
        log.info(f"Devices for robot_id {robot_id}: {rooms}")
        device_list = rooms.get('devices', [])
        device_ids = [device['device_id'] for device in device_list if 'device_id' in device]
        log.info(f"Device IDs for robot_id {robot_id}: {device_ids}")

        TV_app_api = os.getenv("TV_app_api")
        TV_app_websocket = os.getenv("TV_app_websocket")
        TV_app_login_email = os.getenv("TV_app_login_email")
        TV_app_login_password = os.getenv("TV_app_login_password")

        auth_response = sign_in(TV_app_login_email, TV_app_login_password)
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
                    # Send the API request (correcting the payload key to 'url')
                    base_url = os.getenv("base_url")
                    image_path_to_TV = base_url + closest_image_path.replace("\\", "/")
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
        log.error(f"HTTP error retrieving devices for robot_id {robot_id}: {e.detail}")
    except Exception as e:
        log.error(f"Error retrieving devices for robot_id {robot_id}: {e}")

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