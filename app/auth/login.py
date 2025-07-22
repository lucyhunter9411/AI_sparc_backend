from fastapi import HTTPException, status
from app.schemas.auth import LoginRequest
from passlib.context import CryptContext
from fastapi import Depends
from app.core.database import mongo_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def login_user(login_data: LoginRequest, db=Depends(mongo_db)):
    """
    Authenticates a user by email and password.
    """
    async with db as database:
        user = database["auths"].find_one({"email": login_data.email})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        if not verify_password(login_data.password.get_secret_value(), user["password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        # Optionally, generate and return a JWT token here
        return {"message": "Login successful", "user_id": str(user["_id"])}