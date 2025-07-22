from fastapi import HTTPException, status
from app.schemas.auth import RegisterRequest
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

async def register_user(register_data: RegisterRequest, db):
    """
    Registers a new user.
    """
    async with db as database:
        # Check if user already exists
        if database["auths"].find_one({"email": register_data.email}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        # Hash the password
        hashed_password = get_password_hash(register_data.password.get_secret_value())
        # Create user document
        user_doc = {
            "email": register_data.email,
            "password": hashed_password,
        }
        result = database["auths"].insert_one(user_doc)
        return {
            "message": "Registration successful",
            "user_id": str(result.inserted_id)
        }