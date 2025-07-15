from fastapi import APIRouter, Depends, HTTPException, Form, Query
from app.services.classroom_service import get_classroom_service, ClassroomService
from app.schemas.classroom import (
    ClassroomCreateRequest, 
    ClassroomDeleteRequest, 
    ClassroomResponse, 
    ClassroomsListResponse,
    ClassroomDetailResponse
)
from typing import Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classrooms", tags=["classrooms"])

@router.post("/add", response_model=ClassroomResponse)
async def add_classroom(
    request: ClassroomCreateRequest,
    classroom_service: ClassroomService = Depends(get_classroom_service)
):
    """Add a new classroom."""
    try:
        return await classroom_service.add_or_update_classroom(
            classroom_name=request.classroom_name,
            robot_id=request.robot_id,
            device_id=request.device_id
        )
    except Exception as e:
        logger.error(f"Error in add_classroom endpoint: {e}")
        raise

@router.post("/add/form", response_model=ClassroomResponse)
async def add_classroom_form(
    classroom_name: str = Form(...),
    robot_id: str = Form(...),
    device_id: str = Form(...),
    classroom_service: ClassroomService = Depends(get_classroom_service)
):
    """Add a new classroom using form data."""
    try:
        return await classroom_service.add_or_update_classroom(
            classroom_name=classroom_name,
            robot_id=robot_id,
            device_id=device_id
        )
    except Exception as e:
        logger.error(f"Error in add_classroom_form endpoint: {e}")
        raise

@router.put("/update", response_model=ClassroomResponse)
async def update_classroom(
    request: ClassroomCreateRequest,
    classroom_service: ClassroomService = Depends(get_classroom_service)
):
    """Update an existing classroom."""
    try:
        return await classroom_service.add_or_update_classroom(
            classroom_name=request.classroom_name,
            robot_id=request.robot_id,
            device_id=request.device_id
        )
    except Exception as e:
        logger.error(f"Error in update_classroom endpoint: {e}")
        raise

@router.put("/update/form", response_model=ClassroomResponse)
async def update_classroom_form(
    classroom_name: str = Form(...),
    robot_id: str = Form(...),
    device_id: str = Form(...),
    classroom_service: ClassroomService = Depends(get_classroom_service)
):
    """Update an existing classroom using form data."""
    try:
        return await classroom_service.add_or_update_classroom(
            classroom_name=classroom_name,
            robot_id=robot_id,
            device_id=device_id
        )
    except Exception as e:
        logger.error(f"Error in update_classroom_form endpoint: {e}")
        raise

@router.delete("/delete", response_model=ClassroomResponse)
async def delete_classroom(
    classroom_name: str = Query(...),
    classroom_service: ClassroomService = Depends(get_classroom_service)
):
    """Delete a classroom by name."""
    try:
        return await classroom_service.delete_classroom(classroom_name=classroom_name)
    except Exception as e:
        logger.error(f"Error in delete_classroom endpoint: {e}")
        raise

@router.get("/get", response_model=ClassroomDetailResponse)
async def get_classroom(
    classroom_name: str = Query(...),
    classroom_service: ClassroomService = Depends(get_classroom_service)
):
    """Get a specific classroom by name."""
    try:
        classroom = await classroom_service.get_classroom(classroom_name=classroom_name)
        return ClassroomDetailResponse(classroom=classroom)
    except Exception as e:
        logger.error(f"Error in get_classroom endpoint: {e}")
        raise

@router.get("/list", response_model=ClassroomsListResponse)
async def get_all_classrooms(
    classroom_service: ClassroomService = Depends(get_classroom_service)
):
    """Get all classrooms."""
    try:
        return await classroom_service.get_all_classrooms()
    except Exception as e:
        logger.error(f"Error in get_all_classrooms endpoint: {e}")
        raise

@router.get("/list/by-robot", response_model=ClassroomsListResponse)
async def get_classrooms_by_robot(
    robot_id: str = Query(...),
    classroom_service: ClassroomService = Depends(get_classroom_service)
):
    """Get all classrooms for a specific robot."""
    try:
        return await classroom_service.get_classrooms_by_robot(robot_id=robot_id)
    except Exception as e:
        logger.error(f"Error in get_classrooms_by_robot endpoint: {e}")
        raise 