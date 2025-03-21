import numpy as np
import cv2

class EyeGazeDetector:
    def __init__(self):
        # Eye landmarks indices
        self.LEFT_EYE = list(range(36, 42))  # Left eye landmarks
        self.RIGHT_EYE = list(range(42, 48))  # Right eye landmarks
        
        # Thresholds for eye aspect ratio (EAR)
        self.EAR_THRESHOLD = 0.15  # Lowered threshold for eye closure
        self.EAR_RATIO_THRESHOLD = 0.25  # Lowered threshold for eye direction
        
        # Reference points for gaze direction
        self.REFERENCE_POINTS = {
            'center': (0.5, 0.5),
            'left': (0.3, 0.5),
            'right': (0.7, 0.5),
            'up': (0.5, 0.3),
            'down': (0.5, 0.7)
        }
        
        # Glasses detection threshold
        self.GLASSES_THRESHOLD = 0.8  # Threshold for glasses detection

    def calculate_ear(self, eye_points):
        """Calculate the Eye Aspect Ratio (EAR) for a set of eye landmarks."""
        # Compute the vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Compute the horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_glasses(self, landmarks):
        """Detect if glasses are present based on eye landmarks."""
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        
        # Calculate eye aspect ratios
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # If both eyes have very consistent EAR, likely wearing glasses
        ear_diff = abs(left_ear - right_ear)
        return ear_diff < self.GLASSES_THRESHOLD

    def get_eye_center(self, eye_points):
        """Calculate the center point of an eye."""
        return np.mean(eye_points, axis=0)

    def calculate_gaze_direction(self, left_eye_points, right_eye_points, frame_width, frame_height):
        """Calculate gaze direction based on eye landmarks."""
        # Calculate eye centers
        left_center = self.get_eye_center(left_eye_points)
        right_center = self.get_eye_center(right_eye_points)
        
        # Calculate average eye center
        eye_center = (left_center + right_center) / 2
        
        # Calculate eye aspect ratios
        left_ear = self.calculate_ear(left_eye_points)
        right_ear = self.calculate_ear(right_eye_points)
        
        # Check if eyes are closed
        if left_ear < self.EAR_THRESHOLD or right_ear < self.EAR_THRESHOLD:
            return "closed"
        
        # Calculate relative position using actual frame dimensions
        relative_x = eye_center[0] / frame_width
        relative_y = eye_center[1] / frame_height
        
        # Determine gaze direction based on relative position
        if abs(relative_x - self.REFERENCE_POINTS['center'][0]) < 0.15:  # Increased tolerance
            if abs(relative_y - self.REFERENCE_POINTS['center'][1]) < 0.15:  # Increased tolerance
                return "center"
            elif relative_y < self.REFERENCE_POINTS['center'][1]:
                return "up"
            else:
                return "down"
        elif relative_x < self.REFERENCE_POINTS['center'][0]:
            return "left"
        else:
            return "right"

    def detect_distraction(self, landmarks, frame_width, frame_height):
        """Detect if the user is distracted based on eye gaze."""
        # Extract eye landmarks
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        
        # Detect if glasses are present
        has_glasses = self.detect_glasses(landmarks)
        
        # Calculate gaze direction
        gaze_direction = self.calculate_gaze_direction(left_eye, right_eye, frame_width, frame_height)
        
        # Consider gaze as distracted if looking away from center
        distracted = gaze_direction not in ["center", "closed"]
        
        return distracted, gaze_direction 