import base64
import cv2
import numpy as np
from flask import Flask, request
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from eye_gaze import EyeGazeDetector
from utils import refine
import os
from dotenv import load_dotenv
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from time import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants for optimization
FRAME_INTERVAL = 1/15  # Process 15 frames per second
TARGET_RESOLUTION = (320, 240)  # Reduced resolution for processing
BATCH_SIZE = 1  # Process one frame at a time

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://computer-vision-pi.vercel.app",  # Production frontend
            "http://localhost:5173",                  # Local development
            "http://127.0.0.1:5173",                   # Alternative local URL
            "http://focuspoint.it.com",
            "https://computer-vision-git-master-ehab20011s-projects.vercel.app",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Initialize Socket.IO with CORS support and compression
socketio = SocketIO(app,
    cors_allowed_origins=[
        "https://computer-vision-pi.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://www.focuspoint.it.com",
        "https://focuspoint.it.com",
        "https://computer-vision-git-master-ehab20011s-projects.vercel.app",
    ],
    async_mode='threading',
    logger=False,
    engineio_logger=False,
    log_output=False,
    compress=True  # Enable compression
)

# Initialize models with memory constraints
try:
    face_detector = FaceDetector("face_detector.onnx")
    mark_detector = MarkDetector("face_landmarks.onnx")
    pose_estimator = None
    eye_gaze_detector = EyeGazeDetector()
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    raise

# store the states for each client sid that joins
client_states = {}

def resize_frame(frame):
    """Resize frame to target resolution while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    target_width, target_height = TARGET_RESOLUTION

    # Calculate aspect ratio
    aspect_ratio = width / height

    # Calculate new dimensions
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize frame
    resized = cv2.resize(frame, (new_width, new_height))

    # Create black canvas of target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate position to center the resized frame
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # Place resized frame on canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

    return canvas

def process_frame_logic(frame, sid):
    try:
        state = client_states[sid]

        if state['pose_estimator'] is None:
            state['pose_estimator'] = PoseEstimator(frame.shape[1], frame.shape[0])

        faces, _ = face_detector.detect(frame, 0.7)
        combined_distraction = False

        if len(faces) > 0:
            face = refine(faces[:1], frame.shape[1], frame.shape[0], 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            if patch.size == 0:
                return "Distracted"

            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            head_distraction, _ = state['pose_estimator'].detect_distraction(marks)
            eye_distraction, gaze_direction = eye_gaze_detector.detect_distraction(frame)

            combined_distraction = head_distraction or eye_distraction
            emit('gaze_direction', {'direction': gaze_direction})

        else:
            combined_distraction = True

        return "Distracted" if combined_distraction else "Focused"

    except Exception as e:
        logger.error(f"Error in process_frame_logic: {e}")
        return "Error"


@socketio.on('connect')
def handle_connect():
    sid = request.sid
    client_states[sid] = {
        'last_frame_time': 0,
        'pose_estimator': None
    }
    print(f'Client connected: {sid}')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    client_states.pop(sid, None)
    print(f'Client disconnected: {sid}')

@socketio.on('frame')
def handle_frame(data):
    sid = request.sid
    state = client_states.get(sid)
    if not state:
        logger.error(f"No state found for sid {sid}")
        return

    current_time = time()
    if current_time - state['last_frame_time'] < FRAME_INTERVAL:
        return

    try:
        img = np.frombuffer(base64.b64decode(data['frame']), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        resized_frame = resize_frame(frame)

        focus_status = process_frame_logic(resized_frame, sid)
        emit('focus_status', {'status': focus_status})
        state['last_frame_time'] = current_time

    except Exception as e:
        logger.error(f"Error in handle_frame: {e}")
        emit('error', {'error': str(e)})

# Health Check Endpoint
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
