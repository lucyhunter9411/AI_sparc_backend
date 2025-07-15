from pydantic import BaseModel, Field
from typing import List, Optional

class ClassroomCreateRequest(BaseModel):
    """Request model for creating/updating a classroom."""
    classroom_name: str = Field(..., description="The classroom name/location")
    robot_id: str = Field(..., description="The robot ID")
    device_id: str = Field(..., description="The device (TV) ID")

class ClassroomDeleteRequest(BaseModel):
    """Request model for deleting a classroom."""
    classroom_name: str = Field(..., description="The classroom name/location")

class ClassroomResponse(BaseModel):
    """Response model for classroom operations."""
    message: str = Field(..., description="Operation result message")
    success: bool = Field(..., description="Operation success status")

class Classroom(BaseModel):
    """Model for a single classroom."""
    classroom_name: str = Field(..., description="The classroom name/location")
    robot_id: str = Field(..., description="The robot ID")
    device_id: str = Field(..., description="The device (TV) ID")

class ClassroomsListResponse(BaseModel):
    """Response model for listing classrooms."""
    classrooms: List[Classroom] = Field(..., description="List of classrooms")

class ClassroomDetailResponse(BaseModel):
    """Response model for a single classroom."""
    classroom: Classroom = Field(..., description="Classroom details") 