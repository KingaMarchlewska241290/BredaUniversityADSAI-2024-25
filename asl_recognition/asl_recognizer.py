"""
Main ASL Recognition System
Combines hand detection, static letter classification, and dynamic letter tracking
"""

import sys
import os
from pathlib import Path

# Ensure utils can be imported:
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import cv2
import numpy as np
import tensorflow as tf
import json

from utils.hand_detector import HandDetector
from utils.dynamic_classifier import DynamicLetterClassifier


class ASLRecognizer:
    def __init__(self, model_path, class_indices_path):
        """
        Initialize ASL Recognizer
        
        Args:
            model_path: Path to trained model (.h5 file)
            class_indices_path: Path to class indices JSON
        """
        # Loading model:
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Loading class indices:
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        # Creating reverse mapping (index -> letter):
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        
        print(f"Loaded {len(self.class_indices)} classes: {list(self.class_indices.keys())}")
        
        # Initializing hand detector:
        self.hand_detector = HandDetector(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initializing dynamic letter classifier:
        self.dynamic_classifier = DynamicLetterClassifier()
        
        # State tracking:
        self.is_tracking_dynamic = False
        self.last_prediction = None
        self.last_confidence = 0.0
        
        # Dynamic letters that need motion tracking:
        self.dynamic_letters = {'J', 'Z'}
        
    def preprocess_image(self, hand_crop):
        """
        Preprocessing hand crop for model input
        
        Args:
            hand_crop: Cropped hand image (BGR)
            
        Returns:
            preprocessed: Image ready for model (normalized, batched)
        """
        # Converting BGR to RGB:
        rgb_image = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        
        # Normalizing to [0, 1]:
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Adding batch dimension:
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def predict_static_letter(self, hand_crop):
        """
        Predicting static letter using trained model
        
        Args:
            hand_crop: Preprocessed hand image (224x224)
            
        Returns:
            (letter, confidence): Predicted letter and confidence
        """
        # Preprocessing:
        input_image = self.preprocess_image(hand_crop)
        
        # Predicting:
        predictions = self.model.predict(input_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        # Getting letter:
        letter = self.idx_to_class[predicted_idx]
        
        return letter, float(confidence)
    
    def process_frame(self, frame):
        """
        Processing single frame for ASL recognition
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            (letter, confidence, annotated_frame): Result and visualization
        """
        # Detecting hands:
        results = self.hand_detector.detect_hands(frame)
        
        # Creating annotated frame:
        annotated_frame = frame.copy()
        
        # No hands detected:
        if not results.multi_hand_landmarks:
            self.is_tracking_dynamic = False
            self.dynamic_classifier.reset()
            return None, 0.0, annotated_frame
        
        # Getting first hand:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Drawing landmarks:
        annotated_frame = self.hand_detector.draw_landmarks(
            annotated_frame, 
            hand_landmarks
        )
        
        # Checking if tracking dynamic letter:
        if self.is_tracking_dynamic:
            # Adding point to trajectory:
            index_tip = hand_landmarks.landmark[8]
            self.dynamic_classifier.add_point(index_tip)
            
            # Drawing trajectory:
            self._draw_trajectory(annotated_frame, hand_landmarks)
            
            # Checking if motion complete:
            if self.dynamic_classifier.is_motion_complete():
                letter, confidence = self.dynamic_classifier.classify_trajectory()
                self.is_tracking_dynamic = False
                self.dynamic_classifier.reset()
                
                if letter:
                    self.last_prediction = letter
                    self.last_confidence = confidence
                    return letter, confidence, annotated_frame
                else:
                    # Failed to classify, fall back to static:
                    pass
            else:
                # Still tracking:
                return "Tracking motion...", 0.0, annotated_frame
        
        # Checking if starting dynamic letter:
        if self.dynamic_classifier.is_starting_pose(hand_landmarks):
            self.is_tracking_dynamic = True
            self.dynamic_classifier.reset()
            self.dynamic_classifier.add_point(hand_landmarks.landmark[8])
            return "Ready for J/Z", 0.0, annotated_frame
        
        # Static letter classification:
        try:
            hand_crop, bbox = self.hand_detector.crop_hand(
                frame, 
                hand_landmarks, 
                target_size=(224, 224)
            )
            
            # Drawing bounding box:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), 
                         (0, 255, 0), 2)
            
            # Predicting:
            letter, confidence = self.predict_static_letter(hand_crop)
            
            self.last_prediction = letter
            self.last_confidence = confidence
            
            return letter, confidence, annotated_frame
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, 0.0, annotated_frame
    
    def _draw_trajectory(self, frame, hand_landmarks):
        """Drawing trajectory points on frame"""
        trajectory = self.dynamic_classifier.tracker.get_trajectory_array()
        if trajectory is None or len(trajectory) < 2:
            return
        
        h, w, _ = frame.shape
        
        # Converting normalized coordinates to pixel coordinates:
        points = []
        for point in trajectory:
            x = int(point[0] * w)
            y = int(point[1] * h)
            points.append((x, y))
        
        # Drawing lines connecting points:
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (255, 0, 255), 3)
        
        # Drawing points:
        for point in points:
            cv2.circle(frame, point, 5, (0, 255, 255), -1)
    
    def run_webcam(self, camera_id=0, confidence_threshold=0.5):
        """
        Running real-time ASL recognition from webcam
        
        Args:
            camera_id: Camera device ID
            confidence_threshold: Minimum confidence to display prediction
        """
        print(f"\nStarting webcam recognition...")
        print(f"Camera: {camera_id}")
        print(f"Confidence threshold: {confidence_threshold}")
        print("\nControls:")
        print("'q' - Quit")
        print("'r' - Reset dynamic tracking")
        print("'s' - Save current frame")
        print("\n" + "="*50)
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Setting camera properties:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Failed to grab frame")
                    continue
                
                # Flipping for mirror effect:
                frame = cv2.flip(frame, 1)
                
                # Processing frame:
                letter, confidence, annotated_frame = self.process_frame(frame)
                
                # Displaying results:
                if letter and confidence >= confidence_threshold:
                    # Main prediction:
                    text = f"{letter}"
                    conf_text = f"{confidence:.1%}"
                    
                    # Drawing background box for text:
                    cv2.rectangle(annotated_frame, (10, 10), (300, 100), 
                                (0, 0, 0), -1)
                    cv2.rectangle(annotated_frame, (10, 10), (300, 100), 
                                (0, 255, 0), 2)
                    
                    # Drawing text:
                    cv2.putText(annotated_frame, text, (20, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
                    cv2.putText(annotated_frame, conf_text, (20, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Instructions:
                cv2.putText(annotated_frame, "Press 'q' to quit", 
                           (10, annotated_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Showing frame:
                cv2.imshow('ASL Recognition', annotated_frame)
                
                # Handle key presses:
                key = cv2.waitKey(5) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.is_tracking_dynamic = False
                    self.dynamic_classifier.reset()
                    print("Reset dynamic tracking")
                elif key == ord('s'):
                    filename = f"asl_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Saved frame to {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            print("\nWebcam closed")
    
    def test_image(self, image_path):
        """
        Testing recognition on a single image
        
        Args:
            image_path: Path to test image
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        letter, confidence, annotated_frame = self.process_frame(frame)
        
        print(f"\nPrediction: {letter}")
        print(f"Confidence: {confidence:.2%}")
        
        # Displaying result:
        if letter:
            cv2.putText(annotated_frame, f"{letter} ({confidence:.2%})", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imshow('Test Image', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
