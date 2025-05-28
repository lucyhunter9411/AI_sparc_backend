# app/websockets/connection_manager.py
"""
ConnectionManager
=================

Keeps track of every active WebSocket, grouped first by *robot_id*
and optionally tagged with a *role* such as “audio”, “speech”, “vision”.

Typical FastAPI usage
---------------------
    mgr: ConnectionManager = Depends(get_conn_mgr)

    await mgr.connect(robot_id, ws)        # on socket open
    ...
    mgr.tag(ws, "speech")                  # when ‘register’ arrives
    ...
    for a in mgr.sockets_by_role(robot_id, "audio"):
        await a.send_json({...})           # fan-out to audio clients
    ...
    mgr.disconnect(robot_id, ws)           # in the finally block
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, Set

from fastapi.websockets import WebSocket


class ConnectionManager:                          
    # ─────────────────────────── lifecycle ────────────────────────────
    def __init__(self) -> None:
        # robot_id  →  set[WebSocket]
        self._active: Dict[str, Set[WebSocket]] = defaultdict(set)
        # WebSocket →  role str
        self._roles: Dict[WebSocket, str] = {}

    async def connect(self, robot_id: str, ws: WebSocket) -> None:
        """Accept *ws* and add it to the pool for *robot_id*."""
        await ws.accept()
        self._active[robot_id].add(ws)

    def disconnect(self, robot_id: str, ws: WebSocket) -> None:
        """Forget *ws* and its role."""
        self._active[robot_id].discard(ws)
        self._roles.pop(ws, None)
        if not self._active[robot_id]:          # drop empty sets
            del self._active[robot_id]

    # ─────────────────────────── role tagging ─────────────────────────
    def tag(self, ws: WebSocket, role: str) -> None:
        """Associate a textual *role* (e.g. 'audio') with this socket."""
        self._roles[ws] = role

    def sockets_by_role(self, robot_id: str, role: str) -> Iterator[WebSocket]:
        """Yield sockets for *robot_id* that were tagged with *role*."""
        for ws in self._active.get(robot_id, []):
            if self._roles.get(ws) == role:
                yield ws

    # ───────────────────────── broadcast helpers ──────────────────────
    async def send(self, robot_id: str, data) -> None:
        """Send JSON-serialisable *data* to **every** socket of *robot_id*."""
        for ws in list(self._active.get(robot_id, [])):
            try:
                await ws.send_json(data)
            except Exception:
                # drop dead sockets silently
                self.disconnect(robot_id, ws)

    async def send_role(self, robot_id: str, role: str, data) -> None:
        """Send *data* only to sockets of *robot_id* that were tagged *role*."""
        for ws in list(self.sockets_by_role(robot_id, role)):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(robot_id, ws)

    async def broadcast_all(self, data) -> None:
        """Send *data* to every socket of every robot."""
        for rid in list(self._active):
            await self.send(rid, data)

    # ─────────────────────────── diagnostics ─────────────────────────
    def stats(self) -> Dict[str, Dict[str, int]]:
        """
        Quick overview → {robot_id: {'total': n, 'by_role': {role: n…}}}
        """
        out: Dict[str, Dict[str, int]] = {}
        for rid, socks in self._active.items():
            by_role: Dict[str, int] = {}
            for ws in socks:
                by_role[self._roles.get(ws, "unTagged")] = (
                    by_role.get(self._roles.get(ws, "unTagged"), 0) + 1
                )
            out[rid] = {"total": len(socks), "by_role": by_role}
        return out
