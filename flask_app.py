import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from eye_gaze import EyeGazeDetector
from utils import refine
import os
import threading
import queue
from dotenv import load_dotenv
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from time import time
from functools import lru_cache
import hashlib
import zlib
import psutil
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
            "http://focuspoint.it.com"
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

# Global variables for frame processing
last_frame_time = 0

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

@lru_cache(maxsize=100)
def cache_landmarks(face_hash):
    """Cache landmarks for similar face regions."""
    return None  # Placeholder for actual landmark caching logic

def process_frame_logic(frame):
    """Process a single frame and return distraction status with memory optimization."""
    global pose_estimator
    
    try:
        if pose_estimator is None:
            pose_estimator = PoseEstimator(frame.shape[1], frame.shape[0])

        # Run face detection with error handling
        faces, _ = face_detector.detect(frame, 0.7)
        combined_distraction = False

        if len(faces) > 0:
            # Process only the first face to save resources
            face = refine(faces[:1], frame.shape[1], frame.shape[0], 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            
            # Extract face patch with bounds checking
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            patch = frame[y1:y2, x1:x2]
            
            # Skip if face patch is too small
            if patch.size == 0:
                return "Distracted"
            
            # Detect landmarks
            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1
                
            head_distraction, _ = pose_estimator.detect_distraction(marks)
            
            eye_distraction, gaze_direction = eye_gaze_detector.detect_distraction(marks, frame.shape[1], frame.shape[0])
            
            # Combine distraction results
            combined_distraction = head_distraction or eye_distraction
            
            # Emit gaze direction with reduced frequency
            if time() % 2 < 0.1:  # Only emit every ~2 seconds
                emit('gaze_direction', {'direction': gaze_direction})
        else:
            combined_distraction = True

        return "Distracted" if combined_distraction else "Focused"
        
    except MemoryError:
        logger.error("Memory error in process_frame_logic")
        return "System busy"
    except Exception as e:
        logger.error(f"Error in process_frame_logic: {e}")
        return "Error"

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    """Handle incoming frame data through WebSocket with optimizations."""
    global last_frame_time
    
    current_time = time()
    
    # Frame rate limiting (15 FPS to reduce resource usage)
    if current_time - last_frame_time < FRAME_INTERVAL:
        return  # Skip frame if too soon
    
    try:
        # Decode the base64-encoded frame
        img = np.frombuffer(base64.b64decode(data['frame']), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # Resize frame for faster processing
        resized_frame = resize_frame(frame)
        
        # Process frame and get focus status
        focus_status = process_frame_logic(resized_frame)
        
        # Emit the focus status back to the client
        emit('focus_status', {'status': focus_status})
            
        # Update last frame time
        last_frame_time = current_time
            
    except Exception as e:
        logger.error(f"Error in handle_frame: {e}")
        emit('error', {'error': str(e)})

# Health Check Endpoint
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)