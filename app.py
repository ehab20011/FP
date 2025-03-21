import base64
import cv2
import numpy as np
import requests
import streamlit as st
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
import os
from dotenv import load_dotenv
import time
import threading
import queue


load_dotenv()

# Constants
IMG_PATH = "image.jpg"
API_KEY = os.getenv("ROBOFLOW-INFERENCE-API-KEY")
DISTANCE_TO_OBJECT = 1000  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = (
    f"http://{os.getenv('CONTAINER-PORT')}:9001/gaze/gaze_detection?api_key=" + API_KEY
)

MAX_YAW_LEFT = -0.5  # Maximum yaw for left (e.g., -45 degrees)
MAX_YAW_RIGHT = 0.5  # Maximum yaw for right (e.g., 45 degrees)
MAX_PITCH_UP = -0.5  # Maximum pitch for up (e.g., -30 degrees)
MAX_PITCH_DOWN = 0.5  # Maximum pitch for down (e.g., 30 degrees)

last_eye_distraction = None


def check_for_distraction(gaze):
    """
    Check if the gaze is out of range (distraction) based on yaw (left-right) and pitch (top-bottom).
    """
    yaw = gaze["yaw"]
    pitch = gaze["pitch"]

    # Check if yaw or pitch exceeds defined distraction thresholds
    if yaw < MAX_YAW_LEFT or yaw > MAX_YAW_RIGHT or pitch < MAX_PITCH_UP or pitch > MAX_PITCH_DOWN:
        return True
    return False

def async_gaze_detection(input_queue, output_queue):
    """Run gaze detection in a separate thread."""
    global last_eye_distraction  # Use the global variable to persist the result

    while True:
        # Wait for a frame from the input queue
        frame = input_queue.get()
        if frame is None:  # Stop the thread if None is sent
            break

        # Perform gaze detection and check for distraction
        gazes = detect_gazes(frame)
        eye_distraction = False
        if gazes:
            for gaze in gazes:
                eye_distraction = check_for_distraction(gaze)

        # Update the global variable
        last_eye_distraction = eye_distraction

        # Send the distraction result to the output queue
        output_queue.put(eye_distraction)

def detect_gazes(frame: np.ndarray):
    """Detect gazes from the inference server."""
    if frame is None or frame.size == 0:
        print("Error: Empty or invalid frame passed to detect_gazes.")
        return []

    try:
        _, img_encode = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(img_encode)
        resp = requests.post(
            GAZE_DETECTION_URL,
            json={
                "api_key": API_KEY,
                "image": {"type": "base64", "value": img_base64.decode("utf-8")},
            },
        )
        resp.raise_for_status()
        gazes = resp.json()[0]["predictions"]
        return gazes
    except Exception as e:
        print(f"Error in detect_gazes: {e}")
        return []

# def draw_gaze(img: np.ndarray, gaze: dict):
#     """Draw gaze direction and keypoints on the image."""
#     face = gaze["face"]
#     x_min = int(face["x"] - face["width"] / 2)
#     x_max = int(face["x"] + face["width"] / 2)
#     y_min = int(face["y"] - face["height"] / 2)
#     y_max = int(face["y"] + face["height"] / 2)
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

#     # Draw gaze arrow
#     _, imgW = img.shape[:2]
#     arrow_length = imgW / 2
#     dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
#     dy = -arrow_length * np.sin(gaze["pitch"])
#     cv2.arrowedLine(
#         img,
#         (int(face["x"]), int(face["y"])),
#         (int(face["x"] + dx), int(face["y"] + dy)),
#         (0, 0, 255),
#         2,
#         cv2.LINE_AA,
#         tipLength=0.18,
#     )

#     # Draw keypoints
#     for keypoint in face["landmarks"]:
#         x, y = int(keypoint["x"]), int(keypoint["y"])
#         cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

#     return img

def main():
    st.title("Distraction Detection App")
    distraction_text_placeholder = st.empty()
    combined_distraction = None
    focused_time_placeholder = st.empty()
    distracted_time_placeholder = st.empty()
    # Queues for threading
    input_queue = queue.Queue(maxsize=1)  # Queue for sending frames to the thread
    output_queue = queue.Queue(maxsize=1)  # Queue for receiving distraction results

    # Threading in attempt to prevent eye detection inference calls from slowing down the application 
    thread = threading.Thread(target=async_gaze_detection, args=(input_queue, output_queue))
    thread.daemon = True
    thread.start()
    video_src = 0  # Webcam index

    cap = cv2.VideoCapture(video_src)

    if not cap.isOpened():
        st.error("Unable to access the webcam. Please make sure it's connected and try again.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    frame_placeholder = st.empty()
    yaw_pitch_placeholder = st.empty()

    focused_time = 0
    distracted_time = 0
    start_time = None
    timer_running = False
    timer_ended = False

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start Timer")
    with col2:
        end_btn = st.button("End Timer")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Run face and pose estimation
        faces, _ = face_detector.detect(frame, 0.7)
        if len(faces) > 0:
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]
            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            head_distraction, pose_vectors = pose_estimator.detect_distraction(marks)

            # Send frames to the gaze detection thread every 15 frames
            if input_queue.empty():
                input_queue.put(frame)

            # Retrieve distraction status from the output queue
            eye_distraction = last_eye_distraction
            if not output_queue.empty():
                eye_distraction = output_queue.get()

            # Combine distraction results
            combined_distraction = head_distraction or eye_distraction

        # Update UI elements
        status_text = "Distracted" if combined_distraction else "Focused"
        distraction_text_placeholder.markdown(
            f"""
            <div style="
                font-size: 48px; 
                font-weight: bold; 
                text-align: center; 
                color: {'red' if status_text == 'Distracted' else 'green'}; 
                background-color: {'#ffcccc' if status_text == 'Distracted' else '#ccffcc'};
                border: 2px solid {'red' if status_text == 'Distracted' else 'green'};
                border-radius: 10px;
                padding: 10px;
                width: {frame_placeholder.width}px; /* Match the image width */
                max-width: 100%; 
                margin: 0 auto; /* Center the text */
            ">
                {status_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Timer logic
        now = time.time()
        if start_btn and not timer_running:
            timer_running = True
            start_time = now

        if timer_running:
            elapsed_time = now - start_time
            if combined_distraction:
                distracted_time += elapsed_time
            else:
                focused_time += elapsed_time
            start_time = now  # Update for the next frame

            focused_time_placeholder.write(f"Focused Time: {focused_time:.2f}s")
            distracted_time_placeholder.write(f"Distracted Time: {distracted_time:.2f}s")

        if end_btn and timer_running:
            timer_running = False
            break

    cap.release()
    input_queue.put(None)  # Stop the thread
    thread.join()


if __name__ == "__main__":
    main()