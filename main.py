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

# AWS-specific imports
import aioredis
import boto3
from botocore.exceptions import ClientError

# ================= CONFIG =================
# Environment-based configuration
SECRET_KEY = os.getenv("SECRET_KEY", "SUPER_SECRET_KEY")
ALGORITHM = "HS256"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./camera.db")
REDIS_URL = os.getenv("REDIS_URL")  # For ElastiCache
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Paths - use /tmp in Lambda/Fargate, local in dev
BASE_DIR = "/tmp" if ENVIRONMENT == "production" else "."
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# AWS Clients
s3_client = boto3.client('s3', region_name=AWS_REGION) if S3_BUCKET else None

# Redis client (initialized in startup)
redis_client = None

# ================= APP SETUP =================
app = FastAPI()

# CORS - allow any origin, credentials FALSE as requested
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed back to False as requested
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (only if not using S3)
if ENVIRONMENT != "production":
    app.mount("/static/images", StaticFiles(directory=UPLOAD_DIR), name="images")

# ================= GLOBAL STATE =================
# Local state (fallback if Redis unavailable)
mjpeg_buffer = bytearray()
mjpeg_lock = asyncio.Lock()
frame_event = asyncio.Event()
latest_frame_path: Optional[str] = None
esp32_connected = False
active_clients: List[WebSocket] = []
esp32_socket: Optional[WebSocket] = None
last_esp32_message = 0

# ================= REDIS STATE MANAGEMENT =================
class RedisStateManager:
    """Manages distributed state using Redis for multi-container deployment"""
    
    def __init__(self):
        self.enabled = redis_client is not None
    
    async def get_buffer(self) -> bytes:
        """Get MJPEG buffer from Redis or local"""
        if self.enabled:
            try:
                data = await redis_client.get("mjpeg_buffer")
                return bytes.fromhex(data) if data else bytes(self.local_buffer) if hasattr(self, 'local_buffer') else b""
            except:
                pass
        async with mjpeg_lock:
            return bytes(mjpeg_buffer)
    
    async def append_buffer(self, data: bytes):
        """Append to MJPEG buffer"""
        if self.enabled:
            try:
                # Store as hex string in Redis with 30s expiry
                await redis_client.setex("mjpeg_buffer", 30, data.hex())
            except Exception as e:
                print(f"Redis buffer error: {e}")
        # Always update local buffer too
        async with mjpeg_lock:
            global mjpeg_buffer
            mjpeg_buffer.extend(data)
            if len(mjpeg_buffer) > 2000000:
                mjpeg_buffer = mjpeg_buffer[-1500000:]
    
    async def set_esp32_status(self, connected: bool, socket_id: Optional[str] = None):
        """Track ESP32 connection"""
        if self.enabled:
            try:
                if connected:
                    await redis_client.hset("esp32_status", mapping={
                        "connected": "1",
                        "socket_id": socket_id or "",
                        "last_seen": str(time.time())
                    })
                    await redis_client.expire("esp32_status", 60)
                else:
                    await redis_client.hset("esp32_status", "connected", "0")
            except Exception as e:
                print(f"Redis ESP32 status error: {e}")
    
    async def get_esp32_status(self) -> dict:
        """Get ESP32 connection status"""
        if self.enabled:
            try:
                status = await redis_client.hgetall("esp32_status")
                return {
                    "connected": status.get("connected") == "1",
                    "socket_id": status.get("socket_id", ""),
                    "last_seen": float(status.get("last_seen", 0))
                }
            except:
                pass
        return {
            "connected": esp32_connected,
            "socket_id": "",
            "last_seen": last_esp32_message
        }
    
    async def set_latest_frame(self, filename: str):
        """Store latest frame reference"""
        if self.enabled:
            try:
                await redis_client.setex("latest_frame", 60, filename)
            except:
                pass
    
    async def get_latest_frame(self) -> Optional[str]:
        """Get latest frame reference"""
        if self.enabled:
            try:
                return await redis_client.get("latest_frame")
            except:
                pass
        return latest_frame_path
    
    async def publish_command(self, command: str):
        """Publish command to all instances via Redis Pub/Sub"""
        if self.enabled:
            try:
                await redis_client.publish("commands", command)
            except:
                pass

state_mgr = RedisStateManager()

# ================= MODEL DOWNLOAD =================
def download_drive_file(file_id, destination):
    """Download from Google Drive with virus check bypass"""
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

# Download from S3 if available (faster in AWS)
def download_from_s3(s3_key: str, destination: str):
    """Try S3 first, fallback to Google Drive"""
    if not s3_client:
        return False
    try:
        s3_client.download_file(S3_BUCKET, f"models/{s3_key}", destination)
        print(f"Downloaded {s3_key} from S3")
        return True
    except ClientError:
        return False

# ================= MODEL PATHS =================
FACE_PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "res10.caffemodel")
EMBED_MODEL_PATH = os.path.join(MODEL_DIR, "openface.t7")

FACE_PROTO_ID = "1eq0loGlVgjR6W8-CZ8Qrbc2fHZxo2207"
FACE_MODEL_ID = "14iQZzrsU_MB8IRLCTZyXmQBjZtyB0lCz"
EMBED_MODEL_ID = "1l7eEf-YJ6cHEw7tZ1EFjgDKjhaVwwhyr"

detector = None
embedder = None

def load_models():
    """Load face detection and embedding models"""
    global detector, embedder
    
    # Try S3 first, then Google Drive
    if not download_from_s3("deploy.prototxt", FACE_PROTO_PATH):
        download_drive_file(FACE_PROTO_ID, FACE_PROTO_PATH)
    if not download_from_s3("res10.caffemodel", FACE_MODEL_PATH):
        download_drive_file(FACE_MODEL_ID, FACE_MODEL_PATH)
    if not download_from_s3("openface.t7", EMBED_MODEL_PATH):
        download_drive_file(EMBED_MODEL_ID, EMBED_MODEL_PATH)

    detector = cv2.dnn.readNetFromCaffe(FACE_PROTO_PATH, FACE_MODEL_PATH)
    embedder = cv2.dnn.readNetFromTorch(EMBED_MODEL_PATH)
    print("Models loaded successfully")

# ================= DB =================
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
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
async def startup():
    global redis_client
    
    # Initialize Redis if URL provided
    if REDIS_URL:
        try:
            redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await redis_client.ping()
            print("Redis connected")
            state_mgr.enabled = True
        except Exception as e:
            print(f"Redis connection failed: {e}")
    
    # Load CV models
    load_models()
    
    # Initialize DB
    db = SessionLocal()
    if not db.query(Command).first():
        db.add(Command(mode="idle"))
        db.commit()
    db.close()
    
    print(f"Server started in {ENVIRONMENT} mode")
    print("WebSocket endpoints: /ws/mjpeg (ESP32), /ws/client (Frontend)")

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()

# ================= WEBSOCKET ENDPOINTS =================

@app.websocket("/ws/mjpeg")
async def websocket_mjpeg(websocket: WebSocket):
    """ESP32 MJPEG streaming endpoint with keepalive handling"""
    global mjpeg_buffer, esp32_connected, esp32_socket, last_esp32_message
    
    print(f"WS connection from {websocket.client}")
    
    await websocket.accept()
    
    instance_id = os.getenv("HOSTNAME", "local-instance")
    esp32_socket = websocket
    esp32_connected = True
    last_esp32_message = time.time()
    
    # Update distributed state
    await state_mgr.set_esp32_status(True, instance_id)
    print(f"ESP32 connected to instance {instance_id}")
    
    try:
        while True:
            # Check if connection is stale (no message for 40 seconds)
            current_time = time.time()
            if current_time - last_esp32_message > 40:
                print("Connection stale, closing")
                break
            
            # Check if another instance took over (distributed mode)
            status = await state_mgr.get_esp32_status()
            if status.get("socket_id") != instance_id and status.get("connected"):
                print("ESP32 connected to different instance")
                break
            
            try:
                # Short timeout to allow periodic stale checks
                message = await asyncio.wait_for(
                    websocket.receive(), 
                    timeout=5.0
                )
                
                last_esp32_message = time.time()
                await state_mgr.set_esp32_status(True, instance_id)
                
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
                        await state_mgr.publish_command("stream")
                        continue
                        
                elif "bytes" in message:
                    data = message["bytes"]
                    
                    # Update distributed buffer
                    await state_mgr.append_buffer(data)
                    
                    # Also update local for immediate processing
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
        await state_mgr.set_esp32_status(False)
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
    """Extract JPEG frames from MJPEG stream"""
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
    """Save frame locally and optionally to S3"""
    global latest_frame_path
    
    filename = f"latest_{int(time.time())}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    
    with open(path, "wb") as f:
        f.write(jpeg_data)
    
    latest_frame_path = path
    
    # Update distributed state
    await state_mgr.set_latest_frame(filename)
    
    # Upload to S3 if configured
    if s3_client and S3_BUCKET:
        try:
            s3_key = f"uploads/{filename}"
            s3_client.upload_file(path, S3_BUCKET, s3_key)
            # Update with S3 URL if needed
        except Exception as e:
            print(f"S3 upload error: {e}")
    
    # Cleanup old local files
    files = sorted(os.listdir(UPLOAD_DIR))
    if len(files) > 50:
        for old_file in files[:-50]:
            try:
                os.remove(os.path.join(UPLOAD_DIR, old_file))
            except:
                pass
    
    # Log to DB
    db = SessionLocal()
    try:
        db.add(ImageLog(filename=filename))
        db.commit()
    finally:
        db.close()

async def broadcast_to_clients(message: str):
    """Broadcast to all connected frontend clients"""
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
    """Set system command mode"""
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
    
    # Send to ESP32 if connected
    if esp32_socket and esp32_connected:
        try:
            await esp32_socket.send_text(mode)
            print(f"Sent {mode} to ESP32")
        except Exception as e:
            print(f"Send failed: {e}")
    
    # Broadcast to all frontend clients
    await broadcast_to_clients(f"mode:{mode}")
    # Also publish to Redis for other instances
    await state_mgr.publish_command(f"mode:{mode}")

    return {"mode": mode}

# ================= AUTH ROUTES =================
class UserCreate(BaseModel):
    username: str
    password: str

@app.post("/register")
def register(data: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    if db.query(User).filter_by(username=data.username).first():
        raise HTTPException(400, "User exists")
    db.add(User(username=data.username, password=hash_password(data.password)))
    db.commit()
    return {"msg": "registered"}

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get JWT token"""
    user = db.query(User).filter_by(username=form.username).first()
    if not user or not verify_password(form.password, user.password):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token({"sub": user.username})}

# ================= VIDEO STREAMING =================
@app.get("/video-feed")
async def video_feed():
    """HTTP MJPEG video streaming endpoint"""
    async def generate():
        last_hash = 0
        while True:
            # Try distributed buffer first, fallback to local
            current_buffer = await state_mgr.get_buffer()
            
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
    """Direct passthrough of MJPEG stream"""
    async def passthrough():
        while True:
            current_buffer = await state_mgr.get_buffer()
            if current_buffer:
                yield current_buffer
            await asyncio.sleep(0.05)
    
    return StreamingResponse(
        passthrough(),
        media_type="multipart/x-mixed-replace; boundary=123456789000000000000987654321"
    )

# ================= FACE UTILS =================
def detect_faces(frame):
    """Detect faces in frame using DNN"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    detections = detector.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                faces.append(face)
    return faces

def get_embedding(face):
    """Get face embedding vector"""
    blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(blob)
    return embedder.forward()[0]

def match_face(embedding, db):
    """Match embedding against database"""
    for f in db.query(Face).all():
        stored = np.array(list(map(float, f.embedding.split(","))))
        if np.linalg.norm(stored - embedding) < 0.6:
            return f.name
    return "Unknown"

def get_latest_image_path():
    """Get path to latest captured frame"""
    # Try distributed state first
    if redis_client:
        try:
            import asyncio
            filename = asyncio.get_event_loop().run_until_complete(state_mgr.get_latest_frame())
            if filename:
                return os.path.join(UPLOAD_DIR, filename)
        except:
            pass
    return latest_frame_path

# ================= FACE ENROLL =================
@app.post("/create-embedding")
def create_embedding(
    name: str,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create face embedding from latest frame"""
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
    """Recognize face in latest frame"""
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
    """List recent images"""
    imgs = db.query(ImageLog).order_by(ImageLog.created_at.desc()).limit(100).all()
    return [{"file": i.filename, "time": i.created_at} for i in imgs]

@app.get("/latest-frame")
def latest_frame():
    """Get latest frame image"""
    path = get_latest_image_path()
    if not path or not os.path.exists(path):
        raise HTTPException(404, "No frames available")
    return FileResponse(path, media_type="image/jpeg")

# ================= UPLOAD =================
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload image manually"""
    filename = f"capture_{int(time.time())}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    
    global latest_frame_path
    latest_frame_path = path
    
    # Update distributed state
    await state_mgr.set_latest_frame(filename)
    
    # Upload to S3
    if s3_client and S3_BUCKET:
        try:
            s3_client.upload_file(path, S3_BUCKET, f"uploads/{filename}")
        except Exception as e:
            print(f"S3 upload error: {e}")
    
    db = SessionLocal()
    try:
        db.add(ImageLog(filename=filename))
        db.commit()
    finally:
        db.close()
    
    return {"status": "uploaded", "filename": filename}

# ================= STATUS =================
@app.get("/status")
async def status():
    """Get system status"""
    esp32_status = await state_mgr.get_esp32_status()
    
    return {
        "esp32_connected": esp32_status["connected"] if redis_client else esp32_connected,
        "buffer_size": len(mjpeg_buffer),
        "active_clients": len(active_clients),
        "last_esp32_message": esp32_status["last_seen"] if redis_client else last_esp32_message,
        "environment": ENVIRONMENT,
        "redis_connected": redis_client is not None and state_mgr.enabled
    }

@app.get("/ping")
def ping():
    """Simple health check"""
    return {"status": "alive"}

@app.get("/health")
async def health_check():
    """Deep health check for ALB"""
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": ENVIRONMENT
    }
    
    # Check database
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
        raise HTTPException(503, detail=checks)
    
    # Check Redis if configured
    if redis_client:
        try:
            await redis_client.ping()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"error: {str(e)}"
    
    return checks

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "status": "ok",
        "environment": ENVIRONMENT,
        "endpoints": {
            "websocket_esp32": "/ws/mjpeg",
            "websocket_client": "/ws/client",
            "video": "/video-feed",
            "video_direct": "/video-feed-direct",
            "api": "/ping",
            "health": "/health",
            "status": "/status"
        }
    }
