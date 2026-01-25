"""
Hand detection and preprocessing using MediaPipe
Extracts and crops hand regions from frames for model inference
"""

import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=1,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Hands detector
        
        Args:
            static_image_mode: If False, tracks hands across frames (faster)
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect_hands(self, frame):
        """
        Detect hands in frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            results: MediaPipe detection results
        """
        # Converting BGR to RGB:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processing the frame:
        results = self.hands.process(rgb_frame)
        
        return results
    
    def get_hand_bbox(self, hand_landmarks, frame_shape):
        """
        Calculate bounding box around hand landmarks with padding
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            (x_min, y_min, x_max, y_max): Bounding box coordinates
        """
        h, w, _ = frame_shape
        
        # Getting all landmark coordinates:
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        # Finding bounding box:
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Adding padding (20% of box size):
        width = x_max - x_min
        height = y_max - y_min
        
        padding_x = int(width * 0.2)
        padding_y = int(height * 0.2)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        return x_min, y_min, x_max, y_max
    
    def crop_hand(self, frame, hand_landmarks, target_size=(224, 224)):
        """
        Crop and resize hand region for model input
        
        Args:
            frame: BGR image from OpenCV
            hand_landmarks: MediaPipe hand landmarks
            target_size: Target size for model input (width, height)
            
        Returns:
            cropped_hand: Preprocessed hand image ready for model
        """
        # Getting bounding box:
        x_min, y_min, x_max, y_max = self.get_hand_bbox(hand_landmarks, frame.shape)
        
        # Cropping hand region:
        hand_crop = frame[y_min:y_max, x_min:x_max]
        
        # Resizing to target size:
        hand_resized = cv2.resize(hand_crop, target_size, interpolation=cv2.INTER_AREA)
        
        return hand_resized, (x_min, y_min, x_max, y_max)
    
    def draw_landmarks(self, frame, hand_landmarks):
        """
        Draw hand landmarks on frame
        
        Args:
            frame: BGR image from OpenCV
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            frame: Frame with landmarks drawn
        """
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        return frame
    
    def close(self):
        """Release MediaPipe resources"""
        self.hands.close()
