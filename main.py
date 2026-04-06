# ================= IMPORTS =================
from fastapi import FastAPI, Request, Depends, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel
import os, cv2, numpy as np, time, requests, asyncio
from typing import List, Dict, Optional
import io

# ================= CONFIG =================
SECRET_KEY = "SUPER_SECRET_KEY"
ALGORITHM = "HS256"
DATABASE_URL = "sqlite:///./camera.db"

UPLOAD_DIR = "uploads"
UNKNOWN_DIR = "unknown"
MODEL_DIR = "models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= APP SETUP =================
app = FastAPI()

# CORS first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# ================= GLOBAL STATE =================
mjpeg_buffer = bytearray()
mjpeg_lock = asyncio.Lock()
frame_event = asyncio.Event()
latest_frame_path: Optional[str] = None
esp32_connected = False
active_clients: List[WebSocket] = []
esp32_socket: Optional[WebSocket] = None
last_esp32_message = 0  # Track last message time

# ================= MODEL DOWNLOAD =================
def download_drive_file(file_id, destination):
    if os.path.exists(destination):
        return
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value

    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)

# ================= MODEL PATHS =================
FACE_PROTO_PATH = f"{MODEL_DIR}/deploy.prototxt"
FACE_MODEL_PATH = f"{MODEL_DIR}/res10.caffemodel"
EMBED_MODEL_PATH = f"{MODEL_DIR}/openface.t7"

FACE_PROTO_ID = "1eq0loGlVgjR6W8-CZ8Qrbc2fHZxo2207"
FACE_MODEL_ID = "14iQZzrsU_MB8IRLCTZyXmQBjZtyB0lCz"
EMBED_MODEL_ID = "1l7eEf-YJ6cHEw7tZ1EFjgDKjhaVwwhyr"

detector = None
embedder = None

def load_models():
    global detector, embedder
    download_drive_file(FACE_PROTO_ID, FACE_PROTO_PATH)
    download_drive_file(FACE_MODEL_ID, FACE_MODEL_PATH)
    download_drive_file(EMBED_MODEL_ID, EMBED_MODEL_PATH)

    detector = cv2.dnn.readNetFromCaffe(FACE_PROTO_PATH, FACE_MODEL_PATH)
    embedder = cv2.dnn.readNetFromTorch(EMBED_MODEL_PATH)

# ================= DB =================
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

class Face(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    embedding = Column(String)

class ImageLog(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Command(Base):
    __tablename__ = "command"
    id = Column(Integer, primary_key=True)
    mode = Column(String, default="idle")

Base.metadata.create_all(bind=engine)

# ================= AUTH =================
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(pw): return pwd_context.hash(pw)
def verify_password(pw, hashed): return pwd_context.verify(pw, hashed)

def create_token(data):
    data["exp"] = datetime.utcnow() + timedelta(minutes=60)
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except:
        raise HTTPException(401, "Invalid token")

# ================= DB DEP =================
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ================= STARTUP =================
@app.on_event("startup")
def startup():
    load_models()
    db = SessionLocal()
    if not db.query(Command).first():
        db.add(Command(mode="idle"))
        db.commit()
    db.close()
    print("Server started, WebSocket at /ws/mjpeg")

# ================= WEBSOCKET ENDPOINTS =================

@app.websocket("/ws/mjpeg")
async def websocket_mjpeg(websocket: WebSocket):
    """ESP32 MJPEG streaming endpoint with keepalive handling"""
    global mjpeg_buffer, esp32_connected, esp32_socket, last_esp32_message
    
    print(f"WS connection from {websocket.client}")
    
    await websocket.accept()
    
    esp32_socket = websocket
    esp32_connected = True
    last_esp32_message = time.time()
    print("ESP32 connected")
    
    try:
        while True:
            # Check if connection is stale (no message for 40 seconds)
            if time.time() - last_esp32_message > 40:
                print("Connection stale, closing")
                break
            
            try:
                # Short timeout to allow periodic stale checks
                message = await asyncio.wait_for(
                    websocket.receive(), 
                    timeout=5.0
                )
                
                last_esp32_message = time.time()
                
                if "text" in message:
                    text = message["text"]
                    print(f"ESP32: {text}")
                    
                    # Handle pong response
                    if text == "pong":
                        continue
                    # Handle auto-resume request from ESP32
                    elif text == "stream":
                        # Update command in DB
                        db = SessionLocal()
                        try:
                            cmd = db.query(Command).first()
                            if cmd:
                                cmd.mode = "stream"
                                db.commit()
                        finally:
                            db.close()
                        await broadcast_to_clients("mode:stream")
                        continue
                        
                elif "bytes" in message:
                    data = message["bytes"]
                    
                    async with mjpeg_lock:
                        mjpeg_buffer.extend(data)
                        if len(mjpeg_buffer) > 2000000:
                            mjpeg_buffer = mjpeg_buffer[-1500000:]
                    
                    frames = extract_frames_from_mjpeg(bytes(mjpeg_buffer))
                    if frames:
                        await save_frame_for_processing(frames[-1])
                        
            except asyncio.TimeoutError:
                # Send ping to check if ESP32 is alive
                if esp32_socket and esp32_connected:
                    try:
                        await esp32_socket.send_text("ping")
                    except Exception as e:
                        print(f"Failed to send ping: {e}")
                        break
                continue
                
    except WebSocketDisconnect:
        print("ESP32 disconnected cleanly")
    except Exception as e:
        print(f"WS error: {e}")
    finally:
        esp32_connected = False
        esp32_socket = None
        print("ESP32 connection cleanup complete")

@app.websocket("/ws/client")
async def websocket_client(websocket: WebSocket):
    """Frontend notifications"""
    await websocket.accept()
    active_clients.append(websocket)
    print(f"Frontend connected, total: {len(active_clients)}")
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_clients:
            active_clients.remove(websocket)
        print(f"Frontend disconnected, total: {len(active_clients)}")

# ================= MJPEG PARSING =================
BOUNDARY = b"--123456789000000000000987654321"

def extract_frames_from_mjpeg(data: bytes) -> list:
    frames = []
    parts = data.split(BOUNDARY)
    
    for part in parts:
        if b"Content-Type: image/jpeg" in part:
            header_end = part.find(b"\r\n\r\n")
            if header_end != -1:
                jpeg_data = part[header_end + 4:]
                if jpeg_data.endswith(b"\r\n"):
                    jpeg_data = jpeg_data[:-2]
                if jpeg_data.startswith(b"\r\n"):
                    jpeg_data = jpeg_data[2:]
                if len(jpeg_data) > 100:
                    frames.append(jpeg_data)
    
    return frames

async def save_frame_for_processing(jpeg_data: bytes):
    global latest_frame_path
    
    filename = f"latest_{int(time.time())}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    
    with open(path, "wb") as f:
        f.write(jpeg_data)
    
    latest_frame_path = path
    
    files = sorted(os.listdir(UPLOAD_DIR))
    if len(files) > 50:
        for old_file in files[:-50]:
            try:
                os.remove(os.path.join(UPLOAD_DIR, old_file))
            except:
                pass
    
    db = SessionLocal()
    try:
        db.add(ImageLog(filename=filename))
        db.commit()
    finally:
        db.close()

async def broadcast_to_clients(message: str):
    disconnected = []
    for client in active_clients:
        try:
            await client.send_text(message)
        except:
            disconnected.append(client)
    
    for client in disconnected:
        if client in active_clients:
            active_clients.remove(client)

# ================= COMMAND =================
@app.post("/set-command")
async def set_command(
    mode: str,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    global esp32_socket
    
    if mode not in ["stream", "capture", "idle"]:
        raise HTTPException(400, "Invalid mode")

    cmd = db.query(Command).first()
    if not cmd:
        cmd = Command(mode=mode)
        db.add(cmd)
    else:
        cmd.mode = mode

    db.commit()
    
    if esp32_socket and esp32_connected:
        try:
            await esp32_socket.send_text(mode)
            print(f"Sent {mode} to ESP32")
        except Exception as e:
            print(f"Send failed: {e}")
    
    await broadcast_to_clients(f"mode:{mode}")

    return {"mode": mode}

# ================= AUTH ROUTES =================
class UserCreate(BaseModel):
    username: str
    password: str

@app.post("/register")
def register(data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter_by(username=data.username).first():
        raise HTTPException(400, "User exists")
    db.add(User(username=data.username, password=hash_password(data.password)))
    db.commit()
    return {"msg": "registered"}

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=form.username).first()
    if not user or not verify_password(form.password, user.password):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token({"sub": user.username})}

# ================= VIDEO STREAMING =================
@app.get("/video-feed")
async def video_feed():
    async def generate():
        last_hash = 0
        while True:
            async with mjpeg_lock:
                current_buffer = bytes(mjpeg_buffer)
            
            frames = extract_frames_from_mjpeg(current_buffer)
            
            if frames:
                latest = frames[-1]
                current_hash = hash(latest) & 0xFFFFFFFF
                
                if current_hash != last_hash:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(latest)).encode() + b"\r\n"
                        b"\r\n" + latest + b"\r\n"
                    )
                    last_hash = current_hash
            
            await asyncio.sleep(0.033)
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video-feed-direct")
async def video_feed_direct():
    async def passthrough():
        while True:
            async with mjpeg_lock:
                if mjpeg_buffer:
                    yield bytes(mjpeg_buffer)
            await asyncio.sleep(0.05)
    
    return StreamingResponse(
        passthrough(),
        media_type="multipart/x-mixed-replace; boundary=123456789000000000000987654321"
    )

# ================= FACE UTILS =================
def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104,177,123))
    detector.setInput(blob)
    detections = detector.forward()

    faces = []
    for i in range(detections.shape[2]):
        if detections[0,0,i,2] > 0.6:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            face = frame[y1:y2,x1:x2]
            if face.size > 0:
                faces.append(face)
    return faces

def get_embedding(face):
    blob = cv2.dnn.blobFromImage(face,1/255,(96,96),(0,0,0),swapRB=True)
    embedder.setInput(blob)
    return embedder.forward()[0]

def match_face(embedding, db):
    for f in db.query(Face).all():
        stored = np.array(list(map(float, f.embedding.split(","))))
        if np.linalg.norm(stored - embedding) < 0.6:
            return f.name
    return "Unknown"

def get_latest_image_path():
    return latest_frame_path

# ================= FACE ENROLL =================
@app.post("/create-embedding")
def create_embedding(
    name: str,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    path = get_latest_image_path()
    if not path or not os.path.exists(path):
        raise HTTPException(400, "No image available")

    img = cv2.imread(path)
    if img is None:
        raise HTTPException(400, "Cannot read image")
        
    faces = detect_faces(img)
    if not faces:
        raise HTTPException(400, "No face found")

    emb = get_embedding(faces[0])

    db.add(Face(name=name, embedding=",".join(map(str, emb))))
    db.commit()

    return {"status": "saved", "name": name}

# ================= RECOGNIZE =================
@app.get("/recognize")
def recognize(db: Session = Depends(get_db)):
    path = get_latest_image_path()
    if not path or not os.path.exists(path):
        return {"error": "No images available"}

    img = cv2.imread(path)
    if img is None:
        return {"error": "Cannot read image"}
        
    faces = detect_faces(img)
    if not faces:
        return {"status": "no face"}

    name = match_face(get_embedding(faces[0]), db)

    return {
        "status": "known" if name != "Unknown" else "unknown",
        "name": name
    }

# ================= IMAGES =================
@app.get("/images")
def list_images(db: Session = Depends(get_db)):
    imgs = db.query(ImageLog).order_by(ImageLog.created_at.desc()).limit(100).all()
    return [{"file": i.filename, "time": i.created_at} for i in imgs]

@app.get("/latest-frame")
def latest_frame():
    path = get_latest_image_path()
    if not path or not os.path.exists(path):
        raise HTTPException(404, "No frames available")
    return FileResponse(path, media_type="image/jpeg")

# ================= UPLOAD =================
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    filename = f"capture_{int(time.time())}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    
    global latest_frame_path
    latest_frame_path = path
    
    db = SessionLocal()
    try:
        db.add(ImageLog(filename=filename))
        db.commit()
    finally:
        db.close()
    
    return {"status": "uploaded", "filename": filename}

# ================= STATUS =================
@app.get("/status")
def status():
    return {
        "esp32_connected": esp32_connected,
        "buffer_size": len(mjpeg_buffer),
        "active_clients": len(active_clients),
        "last_esp32_message": last_esp32_message
    }

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": {
            "websocket": "/ws/mjpeg",
            "video": "/video-feed",
            "api": "/ping"
        }
    }
