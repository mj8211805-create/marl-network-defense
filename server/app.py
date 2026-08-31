"""FastAPI Application instance and WebSocket setup for CyberMARL."""

from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import config
from server.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title=config.APP_NAME,
        version=config.APP_VERSION,
        description="Autonomous Multi-Agent Reinforcement Learning and ML for Network Defense"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # Static Files
    static_dir = Path(__file__).resolve().parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def serve_index():
        index_p = static_dir / "index.html"
        if index_p.exists():
            return FileResponse(index_p)
        return {"message": "CyberMARL API running. Static UI not found."}

    # WebSocket connection manager
    active_connections = set()

    @app.websocket("/ws/simulate")
    async def simulation_websocket(websocket: WebSocket):
        await websocket.accept()
        active_connections.add(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_json({"event": "PONG", "data": data})
        except WebSocketDisconnect:
            active_connections.remove(websocket)

    return app


app = create_app()
