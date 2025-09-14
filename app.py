# app.py – mikroserwis FastAPI z trybem MOCK (bez kamerki i bez modelu)
# Działa od razu. Gdy dodasz model ONNX do folderu models/ możesz przejść na tryb ONNX.

import asyncio, os, time, json, sqlite3, threading, math
from typing import Dict, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------- Konfiguracja -----------------
DB_PATH     = os.path.join("data", "telemetry.db")
USER_ID     = "ola"
FPS_TARGET  = 10
MODEL_PATH  = os.path.join("models", "ferplus.onnx")  # jeśli kiedyś wgrasz model
FER_MODE    = os.environ.get("FER_MODE", "auto")       # auto | mock | onnx

# auto: jeśli brak modelu -> mock; jeśli model jest -> onnx
USE_MOCK = FER_MODE == "mock" or (FER_MODE == "auto" and not os.path.exists(MODEL_PATH))

# ----------------- Aplikacja -----------------
app = FastAPI(title="FER Microservice (MOCK/ONNX)", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

os.makedirs("data", exist_ok=True)
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.execute("""
CREATE TABLE IF NOT EXISTS emotions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  user TEXT NOT NULL,
  emotion TEXT NOT NULL,
  arousal REAL NOT NULL,
  confidence REAL NOT NULL
)
""")
_db.commit()

latest_event_lock = threading.Lock()
latest_event: Dict = {}
ws_clients: List[WebSocket] = []

# ----------------- Etykiety i mapy -----------------
FERPLUS_LABELS = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]
EMO_TO_AROUSAL = {"anger":0.85,"fear":0.90,"surprise":0.80,"happiness":0.60,"disgust":0.50,"neutral":0.40,"sadness":0.30,"contempt":0.55}

def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def save_and_broadcast(evt: Dict):
    _db.execute("INSERT INTO emotions(ts,user,emotion,arousal,confidence) VALUES (?,?,?,?,?)",
                (evt["ts"], evt["user"], evt["emotion"], evt["arousal"], evt["confidence"]))
    _db.commit()
    with latest_event_lock:
        latest_event.clear()
        latest_event.update(evt)
        print(f"[FER] {evt['ts']} emo={evt['emotion']} arousal={evt['arousal']} conf={evt['confidence']}")


async def broadcast(evt: Dict):
    dead=[]
    for ws in ws_clients:
        try:
            await ws.send_text(json.dumps(evt))
        except Exception:
            dead.append(ws)
    for ws in dead:
        try: ws_clients.remove(ws)
        except ValueError: pass

# ----------------- Tryb ONNX (włącza się automatycznie, gdy jest model) -----------------
def onnx_available() -> bool:
    return os.path.exists(MODEL_PATH)

def build_onnx_runtime():
    import onnxruntime as ort
    import numpy as np
    import cv2
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return sess, in_name, out_name, face_cascade, np, cv2

def softmax_np(np, x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)

async def capture_loop():
    frame_interval = 1.0 / FPS_TARGET

    # Jeśli ustawiono mock lub brak modelu -> jedziemy w MOCK
    use_mock = USE_MOCK
    if not use_mock and not onnx_available():
        use_mock = True

    if use_mock:
        print("🟡 Start w trybie MOCK (bez kamerki i bez modelu).")
        # Generuj neutral z delikatną zmianą arousal (0.3–0.6) co klatkę
        while True:
            t = time.time()
            arousal = 0.45 + 0.15 * math.sin(t * 0.6)  # płynne falowanie
            evt = {
                "ts": now_iso(),
                "user": USER_ID,
                "emotion": "neutral",
                "arousal": round(max(0.0, min(1.0, arousal)), 3),
                "confidence": 1.0
            }
            save_and_broadcast(evt)
            await broadcast(evt)
            await asyncio.sleep(frame_interval)
    else:
        print("🟢 Start w trybie ONNX (kamera + model).")
        # Załaduj dopiero teraz, by w MOCK nie wymagać zależności
        sess, in_name, out_name, face_cascade, np, cv2 = build_onnx_runtime()

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("⚠️ Kamera niedostępna – powrót do trybu MOCK.")
            while True:
                t = time.time()
                arousal = 0.45 + 0.15 * math.sin(t * 0.6)
                evt = {
                    "ts": now_iso(),
                    "user": USER_ID,
                    "emotion": "neutral",
                    "arousal": round(max(0.0, min(1.0, arousal)), 3),
                    "confidence": 1.0
                }
                save_and_broadcast(evt)
                await broadcast(evt)
                await asyncio.sleep(frame_interval)
        else:
            while True:
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    await asyncio.sleep(0.05)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80)
                )

                if len(faces) == 0:
                    label, conf = "neutral", 0.0
                else:
                    # wybierz największą twarz
                    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
                    pad = int(0.15*max(w,h))
                    x0,y0 = max(0,x-pad), max(0,y-pad)
                    x1,y1 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)

                    # pracujemy na GRAY (już masz 'gray' z całej klatki)
                    face_gray = gray[y0:y1, x0:x1]

                    # wyrównaj histogram (USB cam często to lubi)
                    face_gray = cv2.equalizeHist(face_gray)

                    # ---- PREPROCESS + DIAG dla FER+ ----
                    # niektóre wersje FER+ były trenowane na 0..255 (bez /255). Sprawdzimy wariant 'raw255'.
                    def infer_probs_face(face_gray, scale_mode="raw255"):
                        f = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_LINEAR).astype("float32")
                        if scale_mode != "raw255":
                            f = f / 255.0                 # alternatywnie 0..1 (na wypadek innej kalibracji)
                        f = np.expand_dims(f, (0, 1))     # NCHW: 1x1x64x64
                        out = sess.run([out_name], {in_name: f})[0][0]
                        return out

                    logits_raw = infer_probs_face(face_gray, "raw255")
                    probs_raw = softmax_np(np, logits_raw.astype(np.float64))
                    idx = int(np.argmax(probs_raw))
                    label = FERPLUS_LABELS[idx] if 0 <= idx < len(FERPLUS_LABELS) else "neutral"
                    conf = float(probs_raw[idx])

                    # diagnostyka do konsoli – zobaczysz rozkład prawdopodobieństw
                    print(f"[FER dbg] raw255 top={label} conf={conf:.2f} vec={[round(x,3) for x in probs_raw]}")


                base = EMO_TO_AROUSAL.get(label, 0.4)
                arousal = float(max(0.0, min(1.0, 0.2*base + 0.8*(base*conf))))
                evt = {"ts": now_iso(), "user": USER_ID, "emotion": label,
                       "arousal": round(arousal,3), "confidence": round(conf,3)}

                save_and_broadcast(evt)
                await broadcast(evt)

                elapsed = time.perf_counter() - t0
                await asyncio.sleep(max(0.0, frame_interval - elapsed))

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(capture_loop())

@app.on_event("shutdown")
def on_shutdown():
    _db.close()

class LatestResponse(BaseModel):
    ts: Optional[str] = None
    user: Optional[str] = None
    emotion: Optional[str] = None
    arousal: Optional[float] = None
    confidence: Optional[float] = None

@app.get("/health")
def health():
    return {"status": "ok", "mode": ("MOCK" if USE_MOCK else "AUTO/ONNX")}

@app.get("/latest", response_model=LatestResponse)
def get_latest(user: Optional[str] = None):
    with latest_event_lock:
        return latest_event or LatestResponse().dict()

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        try: ws_clients.remove(websocket)
        except ValueError: pass
