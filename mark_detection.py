"""Human facial landmark detector based on Convolutional Neural Network."""
import os
import boto3
import cv2
import numpy as np
import onnxruntime as ort
from botocore.exceptions import ClientError

def download_from_s3(bucket_name, object_name):
    """Download a file from S3 if it doesn't exist locally."""
    print(f"Attempting to download {object_name} from bucket {bucket_name}")
    s3_client = boto3.client('s3')
    local_path = os.path.join('assets', object_name)
    
    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    print(f"Created assets directory at {os.path.abspath('assets')}")
    
    # Force download from S3
    print(f"Downloading {object_name} from S3...")
    try:
        s3_client.download_file(bucket_name, object_name, local_path)
        file_size = os.path.getsize(local_path)
        print(f"Successfully downloaded {object_name} from S3 to {local_path}")
        print(f"File size: {file_size} bytes")
        
        # Validate file size (ONNX models are typically several MB)
        if file_size < 1024 * 1024:  # Less than 1MB
            print(f"WARNING: File size is suspiciously small ({file_size} bytes)")
            print("This might indicate a corrupted file or download issue")
            
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        raise
        
    return local_path

class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, model_file):
        """Initialize a mark detector.

        Args:
            model_file (str): ONNX model path or S3 object name.
        """
        print(f"Initializing MarkDetector with model file: {model_file}")
        # Always download from S3 to ensure we have a fresh copy
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        print(f"Using bucket name: {bucket_name}")
        model_file = download_from_s3(bucket_name, model_file)
        
        assert os.path.exists(model_file), f"File not found: {model_file}"
        print(f"Model file exists at: {os.path.abspath(model_file)}")
        
        # Validate file size before attempting to load
        file_size = os.path.getsize(model_file)
        if file_size < 1024 * 1024:  # Less than 1MB
            raise ValueError(f"Model file is suspiciously small ({file_size} bytes). This might indicate corruption.")
        
        self._input_size = 128
        print("Creating InferenceSession...")
        # Use only CPU provider since Railway doesn't have GPU support
        self.model = ort.InferenceSession(
            model_file, providers=["CPUExecutionProvider"])
        print("InferenceSession created successfully")

    def _preprocess(self, bgrs):
        """Preprocess the inputs to meet the model's needs.

        Args:
            bgrs (np.ndarray): a list of input images in BGR format.

        Returns:
            tf.Tensor: a tensor
        """
        rgbs = []
        for img in bgrs:
            img = cv2.resize(img, (self._input_size, self._input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgbs.append(img)

        return rgbs

    def detect(self, images):
        """Detect facial marks from an face image.

        Args:
            images: a list of face images.

        Returns:
            marks: the facial marks as a numpy array of shape [Batch, 68*2].
        """
        inputs = self._preprocess(images)
        marks = self.model.run(["dense_1"], {"image_input": inputs})
        return np.array(marks)

    def visualize(self, image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)
