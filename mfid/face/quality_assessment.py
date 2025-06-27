import cv2
import numpy as np
from skimage.filters import laplace
from skimage import img_as_float

class FaceQualityAssessor:
    """
    Assesses the quality of detected face images for optimal recognition
    
    Features:
    - Resolution assessment
    - Blur detection
    - Lighting evaluation
    - Pose angle estimation
    - Overall quality score
    
    A higher score indicates better quality for recognition
    """
    
    def __init__(self):
        """Initialize the FaceQualityAssessor with default parameters"""
        # Thresholds for various quality metrics
        self.min_face_size = 64  # Minimum recommended resolution
        self.optimal_face_size = 224  # Optimal resolution for recognition
        self.blur_threshold = 100  # Laplacian variance threshold for blur detection
        self.lighting_range = (40, 220)  # Good lighting range (min, max pixel values)
        
        # Weights for overall score calculation
        self.weights = {
            'resolution': 0.25,
            'blur': 0.35,
            'lighting': 0.25,
            'pose': 0.15
        }
    
    def assess_quality(self, face_image):
        """
        Assess the quality of a face image for recognition
        
        Args:
            face_image (numpy.ndarray): Face image in BGR format
        
        Returns:
            dict: Dictionary with quality scores and an overall score (0-100)
        """
        if face_image is None or face_image.size == 0:
            return {
                'overall_score': 0,
                'resolution_score': 0,
                'blur_score': 0,
                'lighting_score': 0,
                'pose_score': 0,
                'is_good_quality': False,
                'recommendations': ['Invalid image']
            }
        
        # Convert to grayscale for certain assessments
        if len(face_image.shape) == 3:
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = face_image
        
        # Assess individual quality factors
        resolution_score = self._assess_resolution(face_image)
        blur_score = self._assess_blur(gray_image)
        lighting_score = self._assess_lighting(gray_image)
        pose_score = self._assess_pose(face_image)
        
        # Calculate overall score (weighted average)
        overall_score = (
            self.weights['resolution'] * resolution_score +
            self.weights['blur'] * blur_score +
            self.weights['lighting'] * lighting_score +
            self.weights['pose'] * pose_score
        )
        
        # Determine if the image is good quality (above 60%)
        is_good_quality = overall_score > 60
        
        # Generate recommendations for improvement
        recommendations = self._generate_recommendations(
            resolution_score, blur_score, lighting_score, pose_score
        )
        
        return {
            'overall_score': round(overall_score, 1),
            'resolution_score': round(resolution_score, 1),
            'blur_score': round(blur_score, 1),
            'lighting_score': round(lighting_score, 1),
            'pose_score': round(pose_score, 1),
            'is_good_quality': is_good_quality,
            'recommendations': recommendations
        }
    
    def _assess_resolution(self, image):
        """
        Assess the resolution quality of a face image
        
        Args:
            image (numpy.ndarray): Face image
        
        Returns:
            float: Resolution score (0-100)
        """
        h, w = image.shape[:2]
        min_dimension = min(h, w)
        
        if min_dimension < self.min_face_size:
            # Below minimum threshold
            return (min_dimension / self.min_face_size) * 50
        
        if min_dimension >= self.optimal_face_size:
            # At or above optimal size
            return 100
        
        # Linear scaling between minimum and optimal
        return 50 + (min_dimension - self.min_face_size) / (self.optimal_face_size - self.min_face_size) * 50
    
    def _assess_blur(self, gray_image):
        """
        Detect and assess the level of blur in an image using Laplacian variance
        
        Args:
            gray_image (numpy.ndarray): Grayscale face image
        
        Returns:
            float: Blur score (0-100), higher means less blur
        """
        # Calculate Laplacian variance
        lap_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        if lap_var < 10:  # Extremely blurry
            return 0
        
        if lap_var > self.blur_threshold:
            return 100
        
        # Linear scaling for blur
        return (lap_var / self.blur_threshold) * 100
    
    def _assess_lighting(self, gray_image):
        """
        Assess the lighting conditions in the face image
        
        Args:
            gray_image (numpy.ndarray): Grayscale face image
        
        Returns:
            float: Lighting score (0-100)
        """
        # Calculate histogram
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.flatten() / (gray_image.shape[0] * gray_image.shape[1])
        
        # Calculate mean and standard deviation
        mean_val = np.mean(gray_image)
        std_val = np.std(gray_image)
        
        # Check if the lighting is too dark or too bright
        if mean_val < self.lighting_range[0]:
            # Too dark
            return (mean_val / self.lighting_range[0]) * 50
        
        if mean_val > self.lighting_range[1]:
            # Too bright
            return 100 - min(100, ((mean_val - self.lighting_range[1]) / (255 - self.lighting_range[1])) * 50)
        
        # Assess contrast using standard deviation
        contrast_score = min(100, (std_val / 60) * 100)
        
        # Average of lighting and contrast scores
        return (100 - abs(((mean_val - 128) / 128) * 50) + contrast_score) / 2
    
    def _assess_pose(self, face_image):
        """
        Estimate face pose quality (frontal vs. profile)
        This is a simplified estimation, as accurate pose estimation
        typically requires facial landmarks or a specialized model
        
        Args:
            face_image (numpy.ndarray): Face image
        
        Returns:
            float: Pose score (0-100)
        """
        # This is a simplified method assuming symmetry indicates frontal pose
        # For production, use a proper face alignment/pose estimation model
        if len(face_image.shape) == 3:
            gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = face_image
        
        h, w = gray_image.shape
        left_half = gray_image[:, :w//2]
        right_half = gray_image[:, w//2:]
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize for comparison if needed
        if left_half.shape != right_half_flipped.shape:
            min_w = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_w]
            right_half_flipped = right_half_flipped[:, :min_w]
        
        # Compare the left half with the flipped right half
        # More similar halves indicate a more frontal pose
        similarity = np.mean(cv2.absdiff(left_half, right_half_flipped))
        max_diff = 255  # Maximum possible difference
        
        # Convert to a score (0-100)
        pose_score = 100 - (similarity / max_diff) * 100
        return max(0, min(100, pose_score))
    
    def _generate_recommendations(self, resolution_score, blur_score, lighting_score, pose_score):
        """
        Generate recommendations for improving image quality
        
        Args:
            resolution_score (float): Resolution quality score
            blur_score (float): Blur quality score
            lighting_score (float): Lighting quality score
            pose_score (float): Pose quality score
        
        Returns:
            list: List of recommendation strings
        """
        recommendations = []
        
        if resolution_score < 60:
            recommendations.append("Increase image resolution (closer capture or higher resolution camera)")
        
        if blur_score < 60:
            recommendations.append("Reduce motion blur (stabilize camera and subject)")
        
        if lighting_score < 60:
            recommendations.append("Improve lighting conditions (avoid poor/uneven lighting)")
        
        if pose_score < 60:
            recommendations.append("Capture more frontal face view")
        
        if not recommendations:
            recommendations.append("Good quality image for recognition")
        
        return recommendations


def assess_face_quality(face_image):
    """
    Convenience function to assess the quality of a face image
    
    Args:
        face_image (numpy.ndarray): Face image in BGR format
    
    Returns:
        dict: Dictionary with quality scores and an overall score (0-100)
    """
    assessor = FaceQualityAssessor()
    return assessor.assess_quality(face_image)


def filter_quality_faces(face_images, threshold=60):
    """
    Filter out low-quality face images
    
    Args:
        face_images (list): List of face images
        threshold (int): Quality threshold (0-100)
    
    Returns:
        tuple: (filtered_images, scores) - High quality images and their scores
    """
    assessor = FaceQualityAssessor()
    filtered_images = []
    scores = []
    
    for image in face_images:
        result = assessor.assess_quality(image)
        if result['overall_score'] >= threshold:
            filtered_images.append(image)
            scores.append(result['overall_score'])
    
    return filtered_images, scores 