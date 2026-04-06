from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel
import os, cv2, numpy as np, time, requests

# ================= CONFIG =================
SECRET_KEY = "SUPER_SECRET_KEY" 
ALGORITHM = "HS256"
ESP32_STREAM_URL = "http://192.168.1.50/live"
DATABASE_URL = "sqlite:///./camera.db"

UPLOAD_DIR = "uploads"
UNKNOWN_DIR = "unknown"
MODEL_DIR = "models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= GOOGLE DRIVE DOWNLOAD =================
def download_drive_file(file_id, destination):
    if os.path.exists(destination):
        return

    print(f"Downloading {destination}...")
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

# 🔁 REPLACE WITH YOUR FILE IDs
FACE_PROTO_ID = "1eq0loGlVgjR6W8-CZ8Qrbc2fHZxo2207"
FACE_MODEL_ID = "14iQZzrsU_MB8IRLCTZyXmQBjZtyB0lCz"
EMBED_MODEL_ID = "1l7eEf-YJ6cHEw7tZ1EFjgDKjhaVwwhyr"

# ================= LOAD MODELS =================
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

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= AUTH =================
pwd_context = CryptContext(schemes=["bcrypt"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(pw):
    return pwd_context.hash(pw)

def verify_password(pw, hashed):
    return pwd_context.verify(pw, hashed)

def create_token(data):
    data["exp"] = datetime.utcnow() + timedelta(minutes=60)
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except:
        raise HTTPException(401, "Invalid token")

# ================= MODELS =================
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

# ================= DB DEP =================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ================= INIT =================
@app.on_event("startup")
def startup():
    load_models()

    db = SessionLocal()
    if not db.query(Command).first():
        db.add(Command(mode="idle"))
        db.commit()
    db.close()

# ================= REQUEST SCHEMAS =================
class UserCreate(BaseModel):
    username: str
    password: str

# ================= AUTH ROUTES =================
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

# ================= FACE UTILS =================
def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    detector.setInput(blob)
    detections = detector.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                faces.append((face, (x1, y1, x2, y2)))
    return faces

def get_embedding(face):
    blob = cv2.dnn.blobFromImage(face, 1/255, (96, 96), (0,0,0), swapRB=True)
    embedder.setInput(blob)
    return embedder.forward()[0]

def match_face(embedding, db):
    for f in db.query(Face).all():
        stored = np.array(list(map(float, f.embedding.split(","))))
        if np.linalg.norm(stored - embedding) < 0.6:
            return f.name
    return "Unknown"

# ================= SAFE FILE HELPER =================
def get_latest_image():
    files = sorted(os.listdir(UPLOAD_DIR))
    if not files:
        return None
    return os.path.join(UPLOAD_DIR, files[-1])

# ================= ROUTES =================
@app.post("/upload")
async def upload(request: Request, db: Session = Depends(get_db)):
    body = await request.body()
    filename = f"{int(time.time())}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        f.write(body)

    db.add(ImageLog(filename=filename))
    db.commit()
    return {"file": filename}

@app.post("/create-embedding")
def create_embedding(name: str, db: Session = Depends(get_db)):
    path = get_latest_image()
    if not path:
        return {"error": "No images"}

    img = cv2.imread(path)
    faces = detect_faces(img)

    if not faces:
        return {"error": "No face"}

    emb = get_embedding(faces[0][0])
    db.add(Face(name=name, embedding=",".join(map(str, emb))))
    db.commit()

    return {"msg": f"{name} saved"}

@app.get("/recognize")
def recognize(db: Session = Depends(get_db)):
    path = get_latest_image()
    if not path:
        return {"error": "No images"}

    img = cv2.imread(path)
    faces = detect_faces(img)

    if not faces:
        return {"status": "no face"}

    name = match_face(get_embedding(faces[0][0]), db)
    return {"status": "known" if name != "Unknown" else "unknown", "name": name}

# ================= STREAM =================
def gen_frames(db):
    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    last_save = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        for face, (x1,y1,x2,y2) in detect_faces(frame):
            name = match_face(get_embedding(face), db)

            if name == "Unknown" and time.time() - last_save > 5:
                cv2.imwrite(os.path.join(UNKNOWN_DIR, f"{time.time()}.jpg"), frame)
                last_save = time.time()

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
def video_feed(db: Session = Depends(get_db)):
    return StreamingResponse(gen_frames(db),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/set-command")
def set_command(mode: str, db: Session = Depends(get_db)):
    if mode not in ["stream", "capture", "idle"]:
        raise HTTPException(400, "Invalid mode")

    cmd = db.query(Command).first()
    if not cmd:
        cmd = Command(mode=mode)
        db.add(cmd)
    else:
        cmd.mode = mode

    db.commit()
    return {"mode": mode}
    
@app.get("/camera/command")
def get_command(db: Session = Depends(get_db)):
    cmd = db.query(Command).first()
    return cmd.mode if cmd else "idle"

@app.get("/ping")
def ping():
    return {"status": "alive"}
