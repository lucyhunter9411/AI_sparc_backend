from fastapi import HTTPException, Depends
from app.api.deps import get_db
from app.services.classroom_service import get_classroom_service, ClassroomService
import logging

logger = logging.getLogger(__name__)

async def get_classrooms(robot_id: str, db=Depends(get_db)):
    """
    Legacy function for backward compatibility.
    Use ClassroomService.get_classrooms_by_robot() for new code.
    """
    classroom_service = ClassroomService(db)
    return await classroom_service.get_classrooms_by_robot(robot_id)

# Backward compatibility alias
async def get_rooms(robot_id: str, db=Depends(get_db)):
    """
    Legacy function for backward compatibility.
    Use get_classrooms() for new code.
    """
    return await get_classrooms(robot_id, db)