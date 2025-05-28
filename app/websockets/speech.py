from fastapi import APIRouter, WebSocket, Depends
from starlette.websockets import WebSocketDisconnect
from app.api.deps import get_conn_mgr
from app.websockets.connection_manager import ConnectionManager
from app.schemas.ws import WSMessage
import time, logging

router = APIRouter()
log = logging.getLogger(__name__)

@router.websocket("/ws/{robot_id}/speech")
async def speech_ws(
        ws: WebSocket,
        robot_id: str,
        mgr: ConnectionManager = Depends(get_conn_mgr),
):
    await mgr.connect(robot_id, ws)
    log.info("🗣️  speech client connected for %s", robot_id)

    try:
        while True:
            msg = WSMessage.model_validate(await ws.receive_json())
            if msg.type == "stt_result":
                user_text = msg.data["text"]

                # ── your existing LLM/RAG + TTS pipeline ──
                tts_b64 = await build_tts_for(user_text)

                # send clip ONLY to audio sockets for this robot
                for audio_ws in mgr._active.get(robot_id, []):
                    if "/audio" in audio_ws.url.path:
                        await audio_ws.send_json(
                            WSMessage(
                                type="audio",
                                data={"audio": tts_b64},
                                ts=time.time(),
                            ).model_dump()
                        )
    except WebSocketDisconnect:
        pass
    finally:
        mgr.disconnect(robot_id, ws)
