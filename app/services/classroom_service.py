from fastapi import HTTPException, Depends
from app.api.deps import get_db
from app.schemas.classroom import ClassroomResponse, ClassroomsListResponse, Classroom
import logging
from typing import Dict, List, Optional
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class ClassroomService:
    """Service class for managing classroom operations."""
    
    def __init__(self, db):
        self.db = db
    
    def _validate_inputs(self, classroom_name: str, robot_id: str, device_id: str) -> None:
        """Validate input parameters."""
        if not classroom_name or not classroom_name.strip():
            raise HTTPException(status_code=400, detail="Classroom name cannot be empty")
        if not robot_id or not robot_id.strip():
            raise HTTPException(status_code=400, detail="Robot ID cannot be empty")
        if not device_id or not device_id.strip():
            raise HTTPException(status_code=400, detail="Device ID cannot be empty")
    
    async def add_or_update_classroom(self, classroom_name: str, robot_id: str, device_id: str) -> ClassroomResponse:
        """Add or update a classroom."""
        try:
            # Validate inputs
            self._validate_inputs(classroom_name, robot_id, device_id)
            
            # Create classroom object with validation
            try:
                classroom = Classroom(classroom_name=classroom_name, robot_id=robot_id, device_id=device_id)
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Invalid classroom data: {e}")
            
            # Check if classroom already exists
            existing_classroom = self.db.classrooms.find_one({"classroom_name": classroom_name})
            
            if existing_classroom:
                # Update existing classroom
                result = self.db.classrooms.update_one(
                    {"classroom_name": classroom_name},
                    {"$set": {"robot_id": robot_id, "device_id": device_id}}
                )
                return ClassroomResponse(message="Classroom updated successfully", success=True)
            else:
                # Create new classroom
                classroom_data = classroom.dict()
                result = self.db.classrooms.insert_one(classroom_data)
                if not result.inserted_id:
                    raise HTTPException(status_code=500, detail="Failed to create classroom")
                return ClassroomResponse(message="Classroom created successfully", success=True)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error adding/updating classroom: {e}")
            raise HTTPException(status_code=500, detail="Failed to add/update classroom")
    
    async def delete_classroom(self, classroom_name: str) -> ClassroomResponse:
        """Delete a classroom."""
        try:
            # Validate inputs
            if not classroom_name or not classroom_name.strip():
                raise HTTPException(status_code=400, detail="Classroom name cannot be empty")
            
            # Check if classroom exists
            existing_classroom = self.db.classrooms.find_one({"classroom_name": classroom_name})
            
            if existing_classroom:
                # Delete the classroom
                result = self.db.classrooms.delete_one({"classroom_name": classroom_name})
                if result.deleted_count > 0:
                    return ClassroomResponse(message="Classroom deleted successfully", success=True)
                else:
                    raise HTTPException(status_code=500, detail="Failed to delete classroom")
            else:
                return ClassroomResponse(message="Classroom not found", success=False)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting classroom: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete classroom")
    
    async def get_classroom(self, classroom_name: str) -> Classroom:
        """Get a specific classroom by name."""
        try:
            if not classroom_name or not classroom_name.strip():
                raise HTTPException(status_code=400, detail="Classroom name cannot be empty")
            
            classroom_data = self.db.classrooms.find_one({"classroom_name": classroom_name})
            
            if classroom_data:
                return Classroom(**classroom_data)
            else:
                raise HTTPException(status_code=404, detail="Classroom not found")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving classroom: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve classroom")
    
    async def get_all_classrooms(self) -> ClassroomsListResponse:
        """Get all classrooms."""
        try:
            classrooms_data = list(self.db.classrooms.find())
            classrooms = [Classroom(**data) for data in classrooms_data]
            return ClassroomsListResponse(classrooms=classrooms)
        except Exception as e:
            logger.error(f"Error retrieving all classrooms: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve classrooms")
    
    async def get_classrooms_by_robot(self, robot_id: str) -> ClassroomsListResponse:
        """Get all classrooms for a specific robot."""
        try:
            if not robot_id or not robot_id.strip():
                raise HTTPException(status_code=400, detail="Robot ID cannot be empty")
            
            classrooms_data = list(self.db.classrooms.find({"robot_id": robot_id}))
            classrooms = [Classroom(**data) for data in classrooms_data]
            return ClassroomsListResponse(classrooms=classrooms)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving classrooms for robot: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve classrooms")

# Factory function to create classroom service
def get_classroom_service(db=Depends(get_db)) -> ClassroomService:
    return ClassroomService(db) 