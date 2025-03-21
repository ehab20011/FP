"""Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np


class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, image_width, image_height):
        """Init a pose estimator.

        Args:
            image_width (int): input image width
            image_height (int): input image height
        """
        self.size = (image_height, image_width)
        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])

    def _get_full_model_points(self, filename='assets/model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def solve(self, points):
        """Solve pose with all the 68 image points
        Args:
            points (np.ndarray): points on image.

        Returns:
            Tuple: (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def visualize(self, image, pose, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        rotation_vector, translation_vector = pose
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axes(self, img, pose):
        R, t = pose
        img = cv2.drawFrameAxes(img, self.camera_matrix,
                                self.dist_coeefs, R, t, 30)

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)

        x = self.model_points_68[:, 0]
        y = self.model_points_68[:, 1]
        z = self.model_points_68[:, 2]

        ax.scatter(x, y, z)
        ax.axis('square')
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()

    ###
    # yhm : from chat gpt to detect distraction
    ###
    def rotation_matrix_to_angles(self, rotation_vector):
        """Convert rotation vector to pitch, yaw, and roll angles."""
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0

        return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

    def is_distracted(self, rotation_vector):
        """Determine if the user is distracted based on head pose angles."""
        pitch, yaw, roll = self.rotation_matrix_to_angles(rotation_vector)
        
        # Define thresholds (adjust based on further testing)
        pitch_threshold = (-20, 40)  # up and down
        yaw_threshold = (-35, 30)     # left right
        roll_threshold = (-180, 180) # Centered around -180 degree roll
        # Check if head is roughly considered 'facing forward'
        focus_pitch = pitch_threshold[0] < pitch < pitch_threshold[1]
        focus_yaw = yaw_threshold[0] < yaw < yaw_threshold[1]
        focus_roll = roll_threshold[0] < roll < roll_threshold[1]

        return not (focus_pitch and focus_yaw and focus_roll)
    

    def detect_distraction(self, points):
        """Solve pose and detect distraction status based on pose."""
        rotation_vector, translation_vector = self.solve(points)
        distraction_status = self.is_distracted(rotation_vector)
        return distraction_status, (rotation_vector, translation_vector)
