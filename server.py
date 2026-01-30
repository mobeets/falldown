# uv run python server.py

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
parser.add_argument("--logdir", type=str, default="logs", help="Path to directory to which we will save data")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
parser.add_argument("--port", type=int, default=8000, help="Server port")
parser.add_argument("--no_reload", action="store_true", help="Disable auto-reload")
args, unknown = parser.parse_known_args() # ignore uvicorn args

def ensure_unique_path(path):
    base, ext = os.path.splitext(path)
    n = 1
    unique = path
    while os.path.exists(unique):
        unique = f"{base}-{n}{ext}"
        n += 1
    return unique

# Directory for logs
LOG_DIR = args.logdir
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI()

# Serve static files (index.html, sketch.js, ws_logger.js, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount subfolder static/ at /static
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/configs", StaticFiles(directory=os.path.join(BASE_DIR, "configs")), name="configs")

# Serve your index.html at the root
@app.get("/")
async def get_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = datetime.now().isoformat()
    print(f"[{client_id}] Client connected")

    # No filename yet — wait for the client to send it
    log_file = None

    try:
        while True:
            data = await websocket.receive_text()
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                print("Invalid JSON:", data)
                continue

            # 1) Client chooses filename
            if obj.get("type") == "set_filename":
                requested = obj.get("filename", None)
                if not requested:
                    print("Received set_filename without filename")
                    continue

                # Ensure it ends with .jsonl
                if not requested.endswith(".jsonl"):
                    requested += ".jsonl"

                raw_path = os.path.join(LOG_DIR, requested)
                log_file = ensure_unique_path(raw_path)
                print(f"[{client_id}] Logging to {log_file}")
                continue

            # 2) Ignore data until a filename has been chosen
            if log_file is None:
                print("Received data before filename was set; ignoring")
                continue

            # 3) Append log entry
            with open(log_file, "a") as f:
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
