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
import face_recognition
import time

# ================= CONFIG =================
SECRET_KEY = "SUPER_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

ESP32_STREAM_URL = "http://192.168.1.50/live"  # 🔥 CHANGE THIS

DATABASE_URL = "sqlite:///./camera.db"

UPLOAD_DIR = "uploads"
UNKNOWN_DIR = "unknown"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ================= DB SETUP =================
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
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# ================= MODELS =================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

class Command(Base):
    __tablename__ = "command"
    id = Column(Integer, primary_key=True)
    mode = Column(String, default="idle")

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
    db = SessionLocal()
    if not db.query(Command).first():
        db.add(Command(mode="idle"))
        db.commit()
    db.close()

# ================= AUTH ROUTES =================
@app.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):
    user = User(username=username, password=hash_password(password))
    db.add(user)
    db.commit()
    return {"msg": "registered"}

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# ================= COMMAND =================
@app.post("/set-command")
def set_command(mode: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    if mode not in ["stream", "capture", "idle"]:
        raise HTTPException(400, "Invalid mode")

    cmd = db.query(Command).first()
    cmd.mode = mode
    db.commit()

    return {"mode": mode}

@app.get("/camera/command")
def get_command(db: Session = Depends(get_db)):
    cmd = db.query(Command).first()
    return cmd.mode if cmd else "idle"

# ================= UPLOAD =================
@app.post("/upload")
async def upload_image(request: Request, db: Session = Depends(get_db)):
    body = await request.body()

    filename = f"{int(datetime.utcnow().timestamp())}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        f.write(body)

    log = ImageLog(filename=filename)
    db.add(log)
    db.commit()

    return {"file": filename}

# ================= EMBEDDING =================
def extract_embedding(path):
    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)
    return encodings[0] if encodings else None

@app.post("/create-embedding")
def create_embedding(name: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    images = os.listdir(UPLOAD_DIR)
    if not images:
        return {"error": "No images"}

    latest = sorted(images)[-1]
    path = os.path.join(UPLOAD_DIR, latest)

    emb = extract_embedding(path)
    if emb is None:
        return {"error": "No face detected"}

    face = Face(name=name, embedding=np.array2string(emb))
    db.add(face)
    db.commit()

    return {"msg": f"{name} saved"}

# ================= RECOGNITION =================
@app.get("/recognize")
def recognize(db: Session = Depends(get_db)):
    images = os.listdir(UPLOAD_DIR)
    if not images:
        return {"error": "No images"}

    latest = sorted(images)[-1]
    path = os.path.join(UPLOAD_DIR, latest)

    emb = extract_embedding(path)
    if emb is None:
        return {"status": "No face"}

    faces = db.query(Face).all()

    for f in faces:
        stored = np.fromstring(f.embedding.strip("[]"), sep=' ')
        match = face_recognition.compare_faces([stored], emb)

        if match[0]:
            return {"status": "known", "name": f.name}

    return {"status": "unknown"}

# ================= LIVE VIDEO WITH AI =================
def gen_frames(db: Session):
    cap = cv2.VideoCapture(ESP32_STREAM_URL)

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Optimize: detect every 5 frames
        if frame_count % 5 == 0:
            face_locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, face_locations)

            faces_db = db.query(Face).all()

            for (top, right, bottom, left), face_enc in zip(face_locations, encodings):
                name = "Unknown"

                for f in faces_db:
                    stored = np.fromstring(f.embedding.strip("[]"), sep=' ')
                    match = face_recognition.compare_faces([stored], face_enc)

                    if match[0]:
                        name = f.name
                        break

                # Save unknown faces
                if name == "Unknown":
                    filename = f"{int(time.time())}.jpg"
                    cv2.imwrite(os.path.join(UNKNOWN_DIR, filename), frame)

                # Draw
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

        time.sleep(0.03)

@app.get("/video_feed")
def video_feed(db: Session = Depends(get_db)):
    return StreamingResponse(
        gen_frames(db),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ================= LIST IMAGES =================
@app.get("/images")
def list_images(db: Session = Depends(get_db)):
    imgs = db.query(ImageLog).all()
    return [{"file": i.filename, "time": i.created_at} for i in imgs]

# ================= HEALTH =================
@app.get("/ping")
def ping():
    return {"status": "alive"}
