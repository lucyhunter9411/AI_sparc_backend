"""
Session State Service - Proper Session Isolation
===============================================

This service replaces the problematic global state management in shared_data.py
with proper session-isolated state management.

Key Features:
- Session-isolated state (no cross-contamination)
- Thread-safe operations
- Automatic cleanup of expired sessions
- Type-safe state management
- Proper resource management
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class SessionContext:
    """Context for a single session with automatic cleanup"""
    robot_id: str
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def touch(self) -> None:
        """Update last accessed time"""
        self.last_accessed = time.time()
    
    def age(self) -> float:
        """Get age of session in seconds"""
        return time.time() - self.created_at
    
    def idle_time(self) -> float:
        """Get idle time in seconds"""
        return time.time() - self.last_accessed

class SessionStateService:
    """
    Thread-safe session state management service that eliminates global state pollution.
    
    Each robot gets its own isolated session state, preventing cross-contamination
    between different robot sessions.
    """
    
    def __init__(self, session_timeout: float = 3600.0):  # 1 hour default timeout
        self._sessions: Dict[str, SessionContext] = {}  # session_key -> SessionContext
        self._robot_sessions: Dict[str, List[str]] = defaultdict(list)  # robot_id -> [session_keys]
        self._lock = threading.RLock()
        self._session_timeout = session_timeout
        self._cleanup_task: Optional[asyncio.Task] = None
        
    def _make_session_key(self, robot_id: str, session_id: str) -> str:
        """Create a unique session key"""
        return f"{robot_id}:{session_id}"
    
    def create_session(self, robot_id: str, session_id: str, initial_data: Optional[Dict[str, Any]] = None) -> SessionContext:
        """
        Create a new session context for the given robot and session.
        
        Args:
            robot_id: The robot identifier
            session_id: The session identifier
            initial_data: Optional initial data for the session
            
        Returns:
            SessionContext: The created session context
        """
        with self._lock:
            session_key = self._make_session_key(robot_id, session_id)
            
            # Clean up existing session if it exists
            if session_key in self._sessions:
                self._cleanup_session_unsafe(session_key)
            
            # Create new session
            session_ctx = SessionContext(
                robot_id=robot_id,
                session_id=session_id,
                data=initial_data or {}
            )
            
            self._sessions[session_key] = session_ctx
            self._robot_sessions[robot_id].append(session_key)
            
            logger.info(f"[{robot_id}:{session_id}] Session created")
            return session_ctx
    
    def get_session(self, robot_id: str, session_id: str) -> Optional[SessionContext]:
        """
        Get an existing session context.
        
        Args:
            robot_id: The robot identifier
            session_id: The session identifier
            
        Returns:
            Optional[SessionContext]: The session context if it exists and is active
        """
        with self._lock:
            session_key = self._make_session_key(robot_id, session_id)
            session_ctx = self._sessions.get(session_key)
            
            if session_ctx and session_ctx.is_active:
                session_ctx.touch()
                return session_ctx
            
            return None
    
    def get_or_create_session(self, robot_id: str, session_id: str, 
                            initial_data: Optional[Dict[str, Any]] = None) -> SessionContext:
        """
        Get existing session or create new one if it doesn't exist.
        
        Args:
            robot_id: The robot identifier
            session_id: The session identifier
            initial_data: Optional initial data for new sessions
            
        Returns:
            SessionContext: The session context
        """
        session_ctx = self.get_session(robot_id, session_id)
        if session_ctx is None:
            session_ctx = self.create_session(robot_id, session_id, initial_data)
        return session_ctx
    
    def set_session_data(self, robot_id: str, session_id: str, key: str, value: Any) -> bool:
        """
        Set data in a session context.
        
        Args:
            robot_id: The robot identifier
            session_id: The session identifier
            key: The data key
            value: The data value
            
        Returns:
            bool: True if successful, False if session doesn't exist
        """
        session_ctx = self.get_session(robot_id, session_id)
        if session_ctx:
            session_ctx.data[key] = value
            return True
        return False
    
    def get_session_data(self, robot_id: str, session_id: str, key: str, default: Any = None) -> Any:
        """
        Get data from a session context.
        
        Args:
            robot_id: The robot identifier
            session_id: The session identifier
            key: The data key
            default: Default value if key doesn't exist
            
        Returns:
            Any: The data value or default
        """
        session_ctx = self.get_session(robot_id, session_id)
        if session_ctx:
            return session_ctx.data.get(key, default)
        return default
    
    def close_session(self, robot_id: str, session_id: str) -> bool:
        """
        Close a session and clean up its resources.
        
        Args:
            robot_id: The robot identifier
            session_id: The session identifier
            
        Returns:
            bool: True if session was closed, False if it didn't exist
        """
        with self._lock:
            session_key = self._make_session_key(robot_id, session_id)
            return self._cleanup_session_unsafe(session_key)
    
    def close_robot_sessions(self, robot_id: str) -> int:
        """
        Close all sessions for a specific robot.
        
        Args:
            robot_id: The robot identifier
            
        Returns:
            int: Number of sessions closed
        """
        with self._lock:
            session_keys = self._robot_sessions.get(robot_id, []).copy()
            closed_count = 0
            
            for session_key in session_keys:
                if self._cleanup_session_unsafe(session_key):
                    closed_count += 1
            
            # Clean up robot session list
            if robot_id in self._robot_sessions:
                del self._robot_sessions[robot_id]
            
            logger.info(f"[{robot_id}] Closed {closed_count} sessions")
            return closed_count
    
    def _cleanup_session_unsafe(self, session_key: str) -> bool:
        """
        Clean up a session (must be called with lock held).
        
        Args:
            session_key: The session key to clean up
            
        Returns:
            bool: True if session was cleaned up, False if it didn't exist
        """
        if session_key not in self._sessions:
            return False
        
        session_ctx = self._sessions[session_key]
        session_ctx.is_active = False
        
        # Remove from sessions dict
        del self._sessions[session_key]
        
        # Remove from robot sessions list
        robot_sessions = self._robot_sessions.get(session_ctx.robot_id, [])
        if session_key in robot_sessions:
            robot_sessions.remove(session_key)
        
        logger.debug(f"[{session_ctx.robot_id}:{session_ctx.session_id}] Session cleaned up")
        return True
    
    async def start_cleanup_task(self) -> None:
        """Start the background cleanup task for expired sessions"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Session cleanup task started")
    
    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Session cleanup task stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for session_key, session_ctx in self._sessions.items():
                if session_ctx.idle_time() > self._session_timeout:
                    expired_keys.append(session_key)
            
            for session_key in expired_keys:
                self._cleanup_session_unsafe(session_key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired sessions")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Dict[str, Any]: Service statistics
        """
        with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "active_robots": len(self._robot_sessions),
                "robot_session_counts": {
                    robot_id: len(sessions) 
                    for robot_id, sessions in self._robot_sessions.items()
                }
            }

# Global service instance
session_state_service = SessionStateService()

# Compatibility functions for gradual migration
def get_robot_session_data(robot_id: str, session_id: str, key: str, default: Any = None) -> Any:
    """Get session data for a robot (compatibility function)"""
    return session_state_service.get_session_data(robot_id, session_id, key, default)

def set_robot_session_data(robot_id: str, session_id: str, key: str, value: Any) -> bool:
    """Set session data for a robot (compatibility function)"""
    return session_state_service.set_session_data(robot_id, session_id, key, value)
