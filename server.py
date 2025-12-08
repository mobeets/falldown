# uv run python server.py --save_name test

import os
import json
import argparse
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime

# ---------------------
# Parse command-line args
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("--save_name", type=str, default="events", help="Name of saved JSON log file")
parser.add_argument("--log_dir", type=str, default="logs", help="Path to directory to which we will save data")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
parser.add_argument("--port", type=int, default=8000, help="Server port")
parser.add_argument("--no_reload", action="store_true", help="Disable auto-reload")
args, unknown = parser.parse_known_args() # ignore uvicorn args

# Directory for logs
SAVE_NAME = args.save_name
LOG_DIR = args.log_dir
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{SAVE_NAME}.jsonl")
if os.path.exists(LOG_FILE):
    print(f"{LOG_FILE} already exists. Aborting.")
    exit(1)

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI()

# Serve static files (index.html, sketch.js, ws_logger.js, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount subfolder static/ at /static
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/assets", StaticFiles(directory=os.path.join(BASE_DIR, "assets")), name="assets")
app.mount("/configs", StaticFiles(directory=os.path.join(BASE_DIR, "configs")), name="configs")

# Serve your index.html at the root
@app.get("/")
async def get_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# WebSocket endpoint for logging
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{datetime.now().isoformat()}"
    print(f"[{client_id}] Client connected")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                print("Invalid JSON received:", data)
                continue
            # Append each event to JSONL
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(obj) + "\n")
    except WebSocketDisconnect:
        print(f"[{client_id}] Client disconnected")

# ------------------------
# Run uvicorn programmatically
# ------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",            # module:attribute
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )
