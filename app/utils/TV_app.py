import requests
from fastapi import HTTPException
import os
from dotenv import load_dotenv
from pathlib import Path
# explicitly point at your .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

TV_app_api = os.getenv("TV_app_api")

def sign_in(username: str, password: str) -> dict:
    form_data = {
        "username": username,
        "password": password
    }
    
    response = requests.post(f"{TV_app_api}/auth/token", data=form_data)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Login failed")
    
    return response.json()