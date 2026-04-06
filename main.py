from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from jose import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
import time

# ================= CONFIG =================
SECRET_KEY = "SUPER_SECRET_KEY"
ALGORITHM = "HS256"

ESP32_STREAM_URL = "http://192.168.1.50/live"

DATABASE_URL = "sqlite:///./camera.db"

UPLOAD_DIR = "uploads"
UNKNOWN_DIR = "unknown"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ================= LOAD MODELS =================

FACE_PROTO = "https://drive.google.com/file/d/1eq0loGlVgjR6W8-CZ8Qrbc2fHZxo2207/view"
FACE_MODEL = "https://drive.google.com/file/d/14iQZzrsU_MB8IRLCTZyXmQBjZtyB0lCz/view"
EMBED_MODEL = "https://drive.google.com/file/d/1l7eEf-YJ6cHEw7tZ1EFjgDKjhaVwwhyr/view"

detector = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
embedder = cv2.dnn.readNetFromTorch(EMBED_MODEL)

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
def init():
    db = SessionLocal()
    if not db.query(Command).first():
        db.add(Command(mode="idle"))
        db.commit()
    db.close()

# ================= AUTH ROUTES =================
@app.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):
    db.add(User(username=username, password=hash_password(password)))
    db.commit()
    return {"msg": "registered"}

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=form.username).first()
    if not user or not verify_password(form.password, user.password):
        raise HTTPException(401, "Invalid credentials")

    token = create_token({"sub": user.username})
    return {"access_token": token}

# ================= COMMAND =================
@app.post("/set-command")
def set_command(mode: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    cmd = db.query(Command).first()
    cmd.mode = mode
    db.commit()
    return {"mode": mode}

@app.get("/camera/command")
def get_command(db: Session = Depends(get_db)):
    return db.query(Command).first().mode

# ================= UPLOAD =================
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

# ================= FACE UTILS =================
def get_embedding(face):
    face_blob = cv2.dnn.blobFromImage(face, 1/255, (96, 96), (0,0,0), swapRB=True)
    embedder.setInput(face_blob)
    return embedder.forward()[0]

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
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                faces.append((face, (x1,y1,x2,y2)))
    return faces

def match_face(embedding, db):
    faces = db.query(Face).all()
    for f in faces:
        stored = np.fromstring(f.embedding, sep=",")
        dist = np.linalg.norm(stored - embedding)
        if dist < 0.6:
            return f.name
    return "Unknown"

# ================= CREATE EMBEDDING =================
@app.post("/create-embedding")
def create_embedding(name: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    latest = sorted(os.listdir(UPLOAD_DIR))[-1]
    path = os.path.join(UPLOAD_DIR, latest)

    img = cv2.imread(path)
    faces = detect_faces(img)

    if not faces:
        return {"error": "No face"}

    emb = get_embedding(faces[0][0])
    db.add(Face(name=name, embedding=",".join(map(str, emb))))
    db.commit()

    return {"msg": f"{name} saved"}

# ================= RECOGNIZE =================
@app.get("/recognize")
def recognize(db: Session = Depends(get_db)):
    latest = sorted(os.listdir(UPLOAD_DIR))[-1]
    img = cv2.imread(os.path.join(UPLOAD_DIR, latest))

    faces = detect_faces(img)
    if not faces:
        return {"status": "no face"}

    emb = get_embedding(faces[0][0])
    name = match_face(emb, db)

    return {"status": "known" if name!="Unknown" else "unknown", "name": name}

# ================= VIDEO STREAM =================
def gen_frames(db):
    cap = cv2.VideoCapture(ESP32_STREAM_URL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for face, (x1,y1,x2,y2) in faces:
            emb = get_embedding(face)
            name = match_face(emb, db)

            if name == "Unknown":
                cv2.imwrite(os.path.join(UNKNOWN_DIR, f"{time.time()}.jpg"), frame)

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

# ================= IMAGES =================
@app.get("/images")
def images(db: Session = Depends(get_db)):
    return db.query(ImageLog).all()

# ================= HEALTH =================
@app.get("/ping")
def ping():
    return {"status": "alive"}
