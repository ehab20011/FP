import base64
import cv2
import numpy as np
import os
import requests

class EyeGazeDetector:
    def __init__(self):
        # API Endpoint
        self.API_KEY = os.getenv("ROBOFLOW-INFERENCE-API-KEY")  # Load from .env file
        self.GAZE_DETECTION_URL = f"http://localhost:9001/gaze/gaze_detection?api_key={self.API_KEY}"  # Update API URL
        
        # Thresholds for distraction
        self.MAX_YAW_LEFT = -0.5
        self.MAX_YAW_RIGHT = 0.5
        self.MAX_PITCH_UP = -0.5
        self.MAX_PITCH_DOWN = 0.5

    def check_for_distraction(self, gaze):
        """Check if gaze is out of range (distraction)."""
        yaw = gaze.get("yaw", 0)
        pitch = gaze.get("pitch", 0)
        return yaw < self.MAX_YAW_LEFT or yaw > self.MAX_YAW_RIGHT or pitch < self.MAX_PITCH_UP or pitch > self.MAX_PITCH_DOWN

    def detect_gaze_from_api(self, frame):
        """Send the image to the API and return gaze detection results."""
        try:
            # Convert frame to base64
            _, img_encoded = cv2.imencode(".jpg", frame)
            img_base64 = base64.b64encode(img_encoded).decode("utf-8")

            # Send request to gaze detection API
            response = requests.post(
                self.GAZE_DETECTION_URL,
                json={"image": {"type": "base64", "value": img_base64}}
            )
            response.raise_for_status()

            # Parse API response
            predictions = response.json()[0]["predictions"]
            return predictions  # List of detected faces with gaze info

        except requests.exceptions.RequestException as e:
            print(f"Error contacting API: {e}")
            return []

    def detect_distraction(self, frame):
        """Detect if the user is distracted using the gaze detection API."""
        gazes = self.detect_gaze_from_api(frame)

        eye_distraction = False
        gaze_direction = "center"

        if gazes:
            for gaze in gazes:
                eye_distraction = self.check_for_distraction(gaze)
                gaze_direction = "left" if gaze["yaw"] < -0.5 else "right" if gaze["yaw"] > 0.5 else "center"

        return eye_distraction, gaze_direction
