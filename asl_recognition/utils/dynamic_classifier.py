"""
Trajectory tracking for dynamic ASL letters (J and Z)
Tracks fingertip motion to distinguish between letters
"""

import time
import numpy as np
from collections import deque


class TrajectoryTracker:
    def __init__(self, max_points=30, motion_threshold=0.01):
        """
        Initialize trajectory tracker
        
        Args:
            max_points: Maximum number of points to store (frames)
            motion_threshold: Minimum velocity to consider motion (normalized coords)
        """
        self.max_points = max_points
        self.motion_threshold = motion_threshold
        self.reset()
        
    def reset(self):
        """Reset trajectory tracking"""
        self.points = deque(maxlen=self.max_points)
        self.start_time = None
        self.last_motion_time = None
        
    def add_point(self, landmark):
        """
        Add point to trajectory
        
        Args:
            landmark: MediaPipe landmark (has x, y, z attributes)
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time()
        self.points.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'time': current_time - self.start_time
        })
        
        # Checking if hand is moving:
        if len(self.points) >= 2:
            velocity = self._calculate_velocity()
            if velocity > self.motion_threshold:
                self.last_motion_time = current_time
                
    def _calculate_velocity(self):
        """Calculate current velocity"""
        if len(self.points) < 2:
            return 0.0
            
        p1 = self.points[-2]
        p2 = self.points[-1]
        
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        dt = p2['time'] - p1['time']
        
        if dt == 0:
            return 0.0
            
        velocity = np.sqrt(dx**2 + dy**2) / dt
        return velocity
    
    def is_motion_complete(self, timeout=1.5, still_duration=0.3):
        """
        Check if motion is complete (hand stopped or timeout)
        
        Args:
            timeout: Maximum duration for gesture (seconds)
            still_duration: Time hand must be still to complete (seconds)
            
        Returns:
            bool: True if motion is complete
        """
        if len(self.points) < 10:  # Need minimum points
            return False
            
        current_time = time.time()
        
        # Checking timeout:
        if self.start_time and (current_time - self.start_time) > timeout:
            return True
            
        # Checking if hand has been still:
        if self.last_motion_time:
            time_still = current_time - self.last_motion_time
            if time_still > still_duration:
                return True
                
        return False
    
    def get_trajectory_array(self):
        """Get trajectory as numpy array"""
        if not self.points:
            return None
        return np.array([[p['x'], p['y']] for p in self.points])
    

class DynamicLetterClassifier:
    def __init__(self):
        """Initialize dynamic letter classifier for J and Z"""
        self.tracker = TrajectoryTracker()
        
    def is_starting_pose(self, hand_landmarks):
        """
        Check if hand is in starting position for J or Z
        Both start with extended index finger, other fingers curled
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            bool: True if in starting pose
        """
        # Getting key landmarks:
        index_tip = hand_landmarks.landmark[8]
        index_mcp = hand_landmarks.landmark[5]  # index knuckle
        middle_tip = hand_landmarks.landmark[12]
        middle_mcp = hand_landmarks.landmark[9]  # middle knuckle
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        wrist = hand_landmarks.landmark[0]
        
        # Index finger should be extended (tip higher than knuckle):
        index_extended = index_tip.y < index_mcp.y
        
        # Other fingers should be curled (tips lower than knuckles):
        middle_curled = middle_tip.y > middle_mcp.y
        ring_curled = ring_tip.y > wrist.y + 0.1
        pinky_curled = pinky_tip.y > wrist.y + 0.1
        
        return index_extended and middle_curled and ring_curled
    
    def classify_trajectory(self):
        """
        Classify trajectory as J or Z based on motion pattern
        
        J: Hook shape - downward then curves left/right
        Z: Zigzag - diagonal down-right, horizontal left, diagonal down-right
        
        Returns:
            (letter, confidence): Predicted letter and confidence
        """
        trajectory = self.tracker.get_trajectory_array()
        
        if trajectory is None or len(trajectory) < 10:
            return None, 0.0
        
        # Calculating direction changes (inflection points):
        directions = self._calculate_directions(trajectory)
        direction_changes = self._count_direction_changes(directions)
        
        # Calculating overall motion pattern:
        total_dx = trajectory[-1][0] - trajectory[0][0]
        total_dy = trajectory[-1][1] - trajectory[0][1]
        
        # Z typically has 2+ direction changes and more horizontal motion
        # J typically has 1 direction change and is more curved
        
        if direction_changes >= 2:
            # Likely Z (zigzag pattern):
            confidence = min(0.85, 0.6 + (direction_changes * 0.1))
            return 'Z', confidence
        elif direction_changes >= 1 and abs(total_dy) > abs(total_dx):
            # Likely J (hook pattern with downward motion):
            confidence = 0.75
            return 'J', confidence
        else:
            # Unclear pattern:
            return None, 0.0
    
    def _calculate_directions(self, trajectory):
        """Calculate direction angles between consecutive points"""
        directions = []
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            angle = np.arctan2(dy, dx)
            directions.append(angle)
        return np.array(directions)
    
    def _count_direction_changes(self, directions, threshold=np.pi/3):
        """
        Count significant direction changes
        
        Args:
            directions: Array of direction angles
            threshold: Minimum angle change to count (radians)
            
        Returns:
            int: Number of direction changes
        """
        if len(directions) < 2:
            return 0
            
        changes = 0
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i-1])
            # Handling wraparound:
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
            
            if angle_diff > threshold:
                changes += 1
                
        return changes
    
    def reset(self):
        """Reset tracker"""
        self.tracker.reset()
    
    def add_point(self, landmark):
        """Add point to tracker"""
        self.tracker.add_point(landmark)
        
    def is_motion_complete(self):
        """Check if motion is complete"""
        return self.tracker.is_motion_complete()
