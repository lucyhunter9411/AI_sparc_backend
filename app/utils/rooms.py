from fastapi import HTTPException, Depends
from app.api.deps import get_db
import logging

logger = logging.getLogger(__name__)

async def get_rooms(robot_id: str, db=Depends(get_db)):
    try:
        # Query the database for the robot_id
        existing_entry = db.devices.find_one({"robot_id": robot_id})

        if existing_entry:
            # Return the list of devices
            return {"devices": existing_entry.get("device", [])}
        else:
            # If no entry is found, return an error
            raise HTTPException(status_code=404, detail="Robot ID not found")

    except Exception as e:
        logger.error(f"Error retrieving devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve devices")