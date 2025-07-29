from fastapi import HTTPException, status
from app.schemas.auth import LoginRequest
from passlib.context import CryptContext
from fastapi import Depends
from app.core.database import mongo_db
import jwt
from datetime import datetime, timedelta
import os

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=45)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def login_user(login_data: LoginRequest, db=Depends(mongo_db)):
    """
    Authenticates a user by email and password.
    """
    async with db as database:
        user = database["auths"].find_one({"email": login_data.email})
        if not user or not verify_password(login_data.password.get_secret_value(), user["password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        access_token = create_access_token(data={"sub": str(user["_id"])})
        return {"access_token": access_token, "token_type": "bearer"}
