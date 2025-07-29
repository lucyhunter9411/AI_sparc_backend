from pydantic import BaseModel, Field, EmailStr, SecretStr
from typing import List, Optional
from bson import ObjectId

class PyObjectId(str):
    """Custom ObjectId type for Pydantic validation."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, values=None): 
        if isinstance(v, ObjectId):
            return str(v)  # Convert ObjectId to string
        if isinstance(v, str) and ObjectId.is_valid(v):
            return v  # Keep valid ObjectId strings
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # This tells Pydantic to treat this as a string in OpenAPI/JSON schema
        return {"type": "string"}

# Chapter Schema
class Chapter(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    title: str = Field(...)
    context: str = Field(None)
    image: Optional[str] = Field(None)  

    class Config:
        json_encoders = {ObjectId: str}
        orm_mode = True

# Section Schema (Image is Optional)
class Section(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    chapter_id: PyObjectId = Field(...)  
    title: str = Field(...)
    context: str = Field(None)
    image: Optional[str] = Field(None)

    class Config:
        json_encoders = {ObjectId: str}
        orm_mode = True

class Content(BaseModel):
    text: str
    image: Optional[str] = None
    time: int
    audio: str
    EnglishText: str
    HindiText: str
    TeluguText: str
    EnglishTime: int
    HindiTime:int
    TeluguTime:int

    class Config:
        # This allows Pydantic to convert the data model to and from MongoDB's BSON format
        json_encoders = {
            ObjectId: str
        }
class Topic(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    lecture_id: str
    title: str
    qna_time: int
    content: List[Content]  # A list of Content objects

    class Config:
        json_encoders = {
            ObjectId: str
        }

class QnA(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    question: str = Field(...)
    answer: str = Field(...)
    model : str = Field(...)
    prompt : str = Field(...)
    class Config:
        json_encoders = {ObjectId: str}
        orm_mode = True

class Prompt(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    name: str = Field(...)
    prompt: str = Field(...)  
    class Config:
        json_encoders = {ObjectId: str}
        orm_mode = True
        
class DeviceList(BaseModel):
    device_id: str
    room_name: str

    class Config:
        # This allows Pydantic to convert the data model to and from MongoDB's BSON format
        json_encoders = {
            ObjectId: str
        }

class Devices(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    robot_id: str = Field(...)
    device: List[DeviceList]
    class Config:
        json_encoders = {ObjectId: str}
        orm_mode = True

class Auth(BaseModel):
    """
    Authentication model for user credentials.
    """
    id: Optional[PyObjectId] = Field(default=None, alias="_id", description="MongoDB ObjectId")
    email: EmailStr = Field(..., description="User email address")
    password: SecretStr = Field(..., description="User password (hashed)")

    class Config:
        json_encoders = {ObjectId: str}
        orm_mode = True

class UserData(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_email: str = Field(...)
    selected_room_id: str = Field(...)
    selected_language : str = Field(...)
    show_conv_state : str = Field(...)
    class Config:
        json_encoders = {ObjectId: str}
        orm_mode = True
