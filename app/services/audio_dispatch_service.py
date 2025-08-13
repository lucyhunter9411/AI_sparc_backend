"""
Audio Dispatch Service - Centralized Audio Management
====================================================

This service solves the critical race condition and state management issues
by providing a centralized, thread-safe audio dispatch mechanism.

Key Features:
- Atomic audio client detection
- Proper synchronization between frontend and audio clients
- Centralized state management
- Race condition prevention
- Consistent audio routing logic
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from fastapi import WebSocket
from app.services.shared_data import get_connected_audio_clients

logger = logging.getLogger(__name__)

class AudioClientStatus(Enum):
    UNKNOWN = "unknown"
    CONNECTED = "connected" 
    DISCONNECTED = "disconnected"
    PING_FAILED = "ping_failed"

@dataclass
class AudioDispatchContext:
    """Context for a single audio dispatch operation"""
    robot_id: str
    content_data: Dict[str, Any]
    frontend_websocket: WebSocket
    audio_client_status: AudioClientStatus = AudioClientStatus.UNKNOWN
    ping_response_time: Optional[float] = None
    dispatch_decision: Optional[str] = None  # "audio_client" or "frontend" or "both"
    
class AudioDispatchService:
    """
    Centralized service for managing audio dispatch decisions and operations.
    
    This service ensures atomic decision-making about where audio should be sent,
    preventing race conditions and ensuring consistent behavior.
    """
    
    def __init__(self):
        self._active_dispatches: Dict[str, AudioDispatchContext] = {}
        self._dispatch_lock = asyncio.Lock()
        self._ping_timeout = 2.0  # 2 second timeout for ping responses
        
    async def dispatch_audio(self, robot_id: str, content_data: Dict[str, Any], 
                           frontend_websocket: WebSocket) -> AudioDispatchContext:
        """
        Main entry point for audio dispatch. Makes atomic decision about where
        audio should be sent and handles the dispatch accordingly.
        
        Args:
            robot_id: The robot identifier
            content_data: The content to be sent (text, audio data, etc.)
            frontend_websocket: The frontend WebSocket connection
            
        Returns:
            AudioDispatchContext with dispatch results
        """
        async with self._dispatch_lock:
            # Create dispatch context
            ctx = AudioDispatchContext(
                robot_id=robot_id,
                content_data=content_data,
                frontend_websocket=frontend_websocket
            )
            
            # Store active dispatch to prevent concurrent operations
            self._active_dispatches[robot_id] = ctx
            
            try:
                # Step 1: Detect audio client status atomically
                await self._detect_audio_client_status(ctx)
                
                # Step 2: Make dispatch decision based on status
                self._make_dispatch_decision(ctx)
                
                # Step 3: Execute dispatch based on decision
                await self._execute_dispatch(ctx)
                
                logger.info(f"[{robot_id}] Audio dispatch completed: {ctx.dispatch_decision} "
                           f"(audio_status: {ctx.audio_client_status.value})")
                
                return ctx
                
            finally:
                # Clean up active dispatch
                self._active_dispatches.pop(robot_id, None)
    
    async def _detect_audio_client_status(self, ctx: AudioDispatchContext) -> None:
        """
        Atomically detect the status of audio clients for this robot.
        Uses ping test with timeout to determine if audio client is responsive.
        """
        start_time = time.time()
        
        try:
            # Get audio client connection
            audio_client = get_connected_audio_clients(ctx.robot_id)
            
            if not audio_client:
                ctx.audio_client_status = AudioClientStatus.DISCONNECTED
                logger.info(f"[{ctx.robot_id}] No audio client found")
                return
            
            # Test audio client responsiveness with ping
            ping_start = time.time()
            await asyncio.wait_for(
                audio_client.send_text(json.dumps({"type": "ping"})),
                timeout=self._ping_timeout
            )
            
            ctx.ping_response_time = time.time() - ping_start
            ctx.audio_client_status = AudioClientStatus.CONNECTED
            
            logger.info(f"[{ctx.robot_id}] Audio client ping successful "
                       f"({ctx.ping_response_time:.3f}s)")
            
        except asyncio.TimeoutError:
            ctx.audio_client_status = AudioClientStatus.PING_FAILED
            logger.warning(f"[{ctx.robot_id}] Audio client ping timeout after {self._ping_timeout}s")
            
        except Exception as e:
            ctx.audio_client_status = AudioClientStatus.PING_FAILED
            logger.warning(f"[{ctx.robot_id}] Audio client ping failed: {e}")
    
    def _make_dispatch_decision(self, ctx: AudioDispatchContext) -> None:
        """
        Make atomic decision about where audio should be dispatched based on
        audio client status and system configuration.
        """
        if ctx.audio_client_status == AudioClientStatus.CONNECTED:
            # Audio client is responsive - send only text/image to frontend
            ctx.dispatch_decision = "audio_client"
            logger.info(f"[{ctx.robot_id}] Decision: Audio → Audio Client, Text/Image → Frontend")
            
        elif ctx.audio_client_status in [AudioClientStatus.DISCONNECTED, AudioClientStatus.PING_FAILED]:
            # No responsive audio client - send audio chunks to frontend
            ctx.dispatch_decision = "frontend"
            logger.info(f"[{ctx.robot_id}] Decision: Audio + Text/Image → Frontend")
            
        else:
            # Unknown status - default to frontend for safety
            ctx.dispatch_decision = "frontend"
            logger.warning(f"[{ctx.robot_id}] Decision: Unknown status, defaulting to Frontend")
    
    async def _execute_dispatch(self, ctx: AudioDispatchContext) -> None:
        """
        Execute the dispatch decision by sending content to appropriate clients.
        """
        if ctx.dispatch_decision == "audio_client":
            await self._dispatch_to_audio_client(ctx)
            await self._dispatch_text_to_frontend(ctx)
            
        elif ctx.dispatch_decision == "frontend":
            await self._dispatch_audio_to_frontend(ctx)
            
        else:
            logger.error(f"[{ctx.robot_id}] Invalid dispatch decision: {ctx.dispatch_decision}")
    
    async def _dispatch_to_audio_client(self, ctx: AudioDispatchContext) -> None:
        """
        Send audio data to the audio client via the dedicated audio websocket endpoint.
        """
        try:
            # Import the global audio dispatch variables
            from app.websockets.lecture import data_to_audio, lecture_to_audio
            
            # Prepare multi-language data structure
            language_text_data = self._prepare_language_data(ctx.content_data)
            
            # Set data for audio websocket to pick up
            data_to_audio[ctx.robot_id] = {
                "data": language_text_data
            }
            
            # Update language selection
            if ctx.robot_id not in lecture_to_audio:
                lecture_to_audio[ctx.robot_id] = {}
            
            # Extract selected language from content data
            selected_language = self._extract_selected_language(ctx.content_data)
            lecture_to_audio[ctx.robot_id]["selectedLanguageName"] = selected_language
            
            logger.info(f"[{ctx.robot_id}] Audio data queued for audio client")
            
        except Exception as e:
            logger.error(f"[{ctx.robot_id}] Failed to dispatch to audio client: {e}")
            # Fallback to frontend dispatch
            await self._dispatch_audio_to_frontend(ctx)
    
    async def _dispatch_text_to_frontend(self, ctx: AudioDispatchContext) -> None:
        """
        Send only text and image data to frontend (no audio chunks).
        """
        try:
            selected_language = self._extract_selected_language(ctx.content_data)
            text_content = ctx.content_data.get(f"{selected_language}Text", "")
            image_content = ctx.content_data.get("image", "")
            
            message = {
                "text": text_content,
                "type": "static",
                "image": image_content,
                "audio_source": "audio_client"  # Indicate audio is handled separately
            }
            
            await ctx.frontend_websocket.send_text(json.dumps(message))
            logger.info(f"[{ctx.robot_id}] Text/image sent to frontend")
            
        except Exception as e:
            logger.error(f"[{ctx.robot_id}] Failed to send text to frontend: {e}")
    
    async def _dispatch_audio_to_frontend(self, ctx: AudioDispatchContext) -> None:
        """
        Send audio chunks directly to frontend websocket.
        """
        try:
            from app.utils.audio import generate_audio_stream
            from app.state.lecture_state_machine import chunk_audio
            import os
            
            # Get chunk size from environment
            CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2097152"))
            
            selected_language = self._extract_selected_language(ctx.content_data)
            text_content = ctx.content_data.get(f"{selected_language}Text", "")
            image_content = ctx.content_data.get("image", "")
            
            # Generate audio stream
            audio_stream = generate_audio_stream(text_content, selected_language)
            audio_stream.seek(0)
            audio_bytes = audio_stream.read()
            
            # Create audio chunks
            audio_chunks = chunk_audio(audio_bytes, CHUNK_SIZE)
            
            # Send chunks to frontend
            for i, chunk in enumerate(audio_chunks):
                chunk_message = {
                    "text": text_content if i == 0 else "",
                    "audio_chunk": chunk,
                    "type": "static",
                    "image": image_content if i == 0 else "",
                    "audio_source": "frontend",  # Indicate audio is included
                    "ts": time.time()
                }
                
                await ctx.frontend_websocket.send_text(json.dumps(chunk_message))
            
            logger.info(f"[{ctx.robot_id}] {len(audio_chunks)} audio chunks sent to frontend")
            
        except Exception as e:
            logger.error(f"[{ctx.robot_id}] Failed to send audio to frontend: {e}")
    
    def _prepare_language_data(self, content_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare multi-language data structure for audio client.
        """
        language_data = {}
        
        # Check for different language versions
        for lang_key in ["EnglishText", "HindiText", "TeluguText"]:
            if lang_key in content_data:
                language_data[lang_key] = content_data[lang_key]
        
        # If no language-specific keys found, try to infer from generic text
        if not language_data:
            text = content_data.get("text", "")
            language_data["EnglishText"] = text  # Default to English
        
        return language_data
    
    def _extract_selected_language(self, content_data: Dict[str, Any]) -> str:
        """
        Extract the selected language from content data.
        """
        # Try to find language-specific text keys
        if "EnglishText" in content_data:
            return "English"
        elif "HindiText" in content_data:
            return "Hindi"
        elif "TeluguText" in content_data:
            return "Telugu"
        else:
            return "English"  # Default
    
    def get_dispatch_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current dispatch operations.
        """
        return {
            "active_dispatches": len(self._active_dispatches),
            "active_robot_ids": list(self._active_dispatches.keys())
        }

# Global instance
audio_dispatch_service = AudioDispatchService()
