"""
EDUCATIONAL ASL SYSTEM
- All 26 letters (A-Z) including dynamic J and Z
- Real-time hand position guidance
- Visual feedback overlay
- Detailed instructions for each letter
"""

import sys
import os
from pathlib import Path
from collections import deque, Counter
import cv2
import numpy as np
import time

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from asl_recognizer import ASLRecognizer


class HandPositionAnalyzer:
    """Analyzes hand position and provides visual guidance"""
    
    def get_finger_states(self, hand_landmarks):
        """Determine which fingers are extended"""
        if not hand_landmarks:
            return None
        
        fingers = {
            'thumb': (4, 2),
            'index': (8, 5),
            'middle': (12, 9),
            'ring': (16, 13),
            'pinky': (20, 17)
        }
        
        states = {}
        for name, (tip_idx, base_idx) in fingers.items():
            tip = hand_landmarks.landmark[tip_idx]
            base = hand_landmarks.landmark[base_idx]
            
            if name == 'thumb':
                states[name] = abs(tip.x - base.x) > 0.06
            else:
                states[name] = tip.y < base.y - 0.04
        
        return states
    
    def draw_finger_guide(self, frame, hand_landmarks, target_letter):
        """Draw visual guide showing which fingers should be up/down"""
        if not hand_landmarks:
            return frame
        
        h, w = frame.shape[:2]
        
        # Target finger states for each letter:
        target_states = {
            'A': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'B': {'thumb': False, 'index': True, 'middle': True, 'ring': True, 'pinky': True},
            'C': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'D': {'thumb': True, 'index': True, 'middle': False, 'ring': False, 'pinky': False},
            'E': {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'F': {'thumb': True, 'index': False, 'middle': True, 'ring': True, 'pinky': True},
            'G': {'thumb': True, 'index': True, 'middle': False, 'ring': False, 'pinky': False},
            'H': {'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            'I': {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': True},
            'K': {'thumb': True, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            'L': {'thumb': True, 'index': True, 'middle': False, 'ring': False, 'pinky': False},
            'M': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'N': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'O': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'P': {'thumb': True, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            'Q': {'thumb': True, 'index': True, 'middle': False, 'ring': False, 'pinky': False},
            'R': {'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            'S': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'T': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'U': {'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            'V': {'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
            'W': {'thumb': False, 'index': True, 'middle': True, 'ring': True, 'pinky': False},
            'X': {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
            'Y': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': True},
        }
        
        if target_letter not in target_states:
            return frame
        
        current = self.get_finger_states(hand_landmarks)
        if not current:
            return frame
        
        target = target_states[target_letter]
        
        # Drawing finger status overlay:
        y_start = h - 120
        box_w = 100
        spacing = 110
        
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        display_names = ['THU', 'IND', 'MID', 'RIN', 'PIN']
        
        for i, (finger, emoji) in enumerate(zip(finger_names, display_names)):
            x = 10 + i * spacing
            
            # Checking if correct:
            is_correct = current[finger] == target[finger]
            color = (0, 255, 0) if is_correct else (0, 100, 255)
            
            # Drawing box:
            cv2.rectangle(frame, (x, y_start), (x + box_w, y_start + 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y_start), (x + box_w, y_start + 100), color, 3)
            
            # Finger name:
            cv2.putText(frame, finger[:3].upper(), (x + 10, y_start + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Status:
            target_text = "UP" if target[finger] else "DOWN"
            current_text = "UP" if current[finger] else "DOWN"
            
            cv2.putText(frame, f"Need: {target_text}", (x + 5, y_start + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Got: {current_text}", (x + 5, y_start + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Checkmark or X:
            symbol = "✓" if is_correct else "✗"
            cv2.putText(frame, symbol, (x + 70, y_start + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return frame


class ComprehensiveEducationalSystem:
    """Complete educational system with all 26 letters"""
    
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.hand_analyzer = HandPositionAnalyzer()
        
        # Practice state:
        self.practice_mode = False
        self.target_letter = None
        self.dynamic_mode_for_jz = False
        self.trajectory_jz = []
        self.jz_start_time = None
        
        # Stats:
        self.letter_stats = {letter: {'attempts': 0, 'successes': 0} 
                            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
        
        # Recognition state:
        self.prediction_history = deque(maxlen=20)
        self.last_stable = None
        self.last_confidence = 0.0
        
        # Complete instructions for ALL 26 letters:
        self.instructions = {
            'A': ['Closed fist', 'Thumb on SIDE'],
            'B': ['Flat hand', '4 fingers UP together', 'Thumb across palm'],
            'C': ['Curved hand', 'Like holding a cup'],
            'D': ['Index finger UP', 'Thumb touches middle/ring fingers'],
            'E': ['Very tight fist', 'Fingers curled DOWN hard'],
            'F': ['Circle with index + thumb', 'Other 3 fingers UP'],
            'G': ['Point sideways with index + thumb'],
            'H': ['Point sideways with index + middle'],
            'I': ['Pinky UP only', 'Other fingers in fist'],
            'J': ['DYNAMIC: Extend PINKY finger', 'Draw J shape: DOWN then hook LEFT'],
            'K': ['Index + middle UP at angle', 'Thumb between them'],
            'L': ['L shape', 'Thumb OUT, index UP'],
            'M': ['3 fingers draped over thumb'],
            'N': ['2 fingers draped over thumb'],
            'O': ['Circle with all fingers touching thumb'],
            'P': ['Like K but pointing DOWN'],
            'Q': ['Point DOWN with index + thumb'],
            'R': ['Index crosses over middle', 'Both fingers UP'],
            'S': ['Fist', 'Thumb ACROSS front (not side)'],
            'T': ['Thumb between index and middle fingers'],
            'U': ['Index + middle UP', 'Together (not spread)'],
            'V': ['Peace sign', 'Index + middle SPREAD apart'],
            'W': ['3 fingers UP', 'Index, middle, AND ring'],
            'X': ['Hook your index finger'],
            'Y': ['Shaka sign', 'Thumb + pinky OUT'],
            'Z': ['DYNAMIC: Point index UP', 'Draw Z: diagonal, horizontal, diagonal'],
        }
    
    def start_practice(self, letter):
        """Start practicing a specific letter"""
        self.practice_mode = True
        self.target_letter = letter.upper()
        self.dynamic_mode_for_jz = (letter in ['J', 'Z'])
        self.trajectory_jz = []
        
        print(f"\n[PRACTICE MODE] Letter {self.target_letter}")
        print(f"How to make '{self.target_letter}':")
        for instruction in self.instructions[self.target_letter]:
            print(f"   {instruction}")
        
        # Extra guidance for J and Z:
        if self.target_letter == 'J':
            print("\nTIP FOR J:")
            print("1. Extend PINKY finger (others curled)")
            print("2. Move DOWN in straight line")
            print("3. At bottom, curve LEFT (make a hook)")
            print("Keep it SMOOTH - one flowing motion")
        elif self.target_letter == 'Z':
            print("\nTIP FOR Z:")
            print("1. Point finger up")
            print("2. Move DIAGONAL down-right")
            print("3. Then move LEFT (horizontal)")
            print("4. Then DIAGONAL down-right again")
            print("Make SHARP angles - zigzag pattern")
    
    def exit_practice(self):
        """Exit practice mode"""
        if self.practice_mode:
            attempts = self.letter_stats[self.target_letter]['attempts']
            successes = self.letter_stats[self.target_letter]['successes']
            if attempts > 0:
                rate = (successes / attempts) * 100
                print(f"\n   {self.target_letter}: {successes}/{attempts} = {rate:.0f}% success")
        
        self.practice_mode = False
        self.target_letter = None
        self.dynamic_mode_for_jz = False
        print("\n[FREE PRACTICE] Back to free practice mode")
    
    def process_frame(self, frame):
        """Process frame with all letter support"""
        # Handling dynamic letters J/Z:
        if self.dynamic_mode_for_jz:
            return self._process_dynamic_jz(frame)
        
        # Static letter recognition:
        letter, confidence, annotated = self.recognizer.process_frame(frame)
        
        # Filtering out non-letters:
        if letter in ['del', 'space', 'nothing', 'J', 'Z']:
            letter = None
        
        if letter and confidence > 0.25:
            self.prediction_history.append(letter)
        
        if len(self.prediction_history) < 10:
            return None, 0.0, annotated, None
        
        # Getting consensus:
        recent = list(self.prediction_history)[-15:]
        counts = Counter(recent)
        
        if not counts:
            return None, 0.0, annotated, None
        
        best_letter = counts.most_common(1)[0][0]
        consensus = counts[best_letter] / len(recent)
        
        feedback = None
        if consensus >= 0.5 and self.practice_mode and self.target_letter:
            self.letter_stats[self.target_letter]['attempts'] += 1
            
            if best_letter == self.target_letter:
                self.letter_stats[self.target_letter]['successes'] += 1
                feedback = ["[OK] PERFECT! Well done!"]
            else:
                feedback = [f"[X] Got {best_letter}, need {self.target_letter}", 
                           "[!] Check the finger guide below"]
        
        if consensus >= 0.5:
            self.last_stable = best_letter
            self.last_confidence = consensus
            return best_letter, consensus, annotated, feedback
        
        return self.last_stable, self.last_confidence, annotated, feedback
    
    def _process_dynamic_jz(self, frame):
        """Handle J and Z motion tracking"""
        results = self.recognizer.hand_detector.detect_hands(frame)
        annotated = frame.copy()
        
        if not results.multi_hand_landmarks:
            return None, 0.0, annotated, ["[X] No hand detected"]
        
        hand_landmarks = results.multi_hand_landmarks[0]
        annotated = self.recognizer.hand_detector.draw_landmarks(annotated, hand_landmarks)
        
        # Checking for starting pose - different for J vs Z:
        if self.target_letter == 'J':
            # J uses pinky finger:
            pinky_tip = hand_landmarks.landmark[20]
            pinky_base = hand_landmarks.landmark[17]
            index_tip = hand_landmarks.landmark[8]
            
            # Pinky extended, others curled:
            is_starting = (pinky_tip.y < pinky_base.y and index_tip.y > pinky_base.y)
            tracking_finger = pinky_tip  # Tracking pinky for J
        else:  # Z or default
            # Z uses index finger:
            index_tip = hand_landmarks.landmark[8]
            index_base = hand_landmarks.landmark[5]
            middle_tip = hand_landmarks.landmark[12]
            
            # Index extended, others curledL
            is_starting = (index_tip.y < index_base.y and middle_tip.y > index_base.y)
            tracking_finger = index_tip  # Tracking index for Z
        
        if not self.trajectory_jz and is_starting:
            # Start tracking:
            self.jz_start_time = time.time()
            self.trajectory_jz.append([tracking_finger.x, tracking_finger.y])
            return None, 0.0, annotated, ["[>] START DRAWING NOW!"]
        
        if self.trajectory_jz:
            # Continue tracking:
            self.trajectory_jz.append([tracking_finger.x, tracking_finger.y])
            
            # Draw trajectory:
            h, w = annotated.shape[:2]
            points = [(int(p[0]*w), int(p[1]*h)) for p in self.trajectory_jz]
            for i in range(1, len(points)):
                cv2.line(annotated, points[i-1], points[i], (255, 0, 255), 4)
            
            elapsed = time.time() - self.jz_start_time
            
            # Checking if complete:
            if elapsed > 2.5:
                result = self._classify_jz_trajectory()
                self.trajectory_jz = []
                
                if result == self.target_letter:
                    self.letter_stats[self.target_letter]['successes'] += 1
                    self.letter_stats[self.target_letter]['attempts'] += 1
                    return result, 0.85, annotated, ["[OK] CORRECT! Great motion!"]
                else:
                    self.letter_stats[self.target_letter]['attempts'] += 1
                    return result, 0.5, annotated, [f"[X] Try again - draw {self.target_letter} shape"]
            
            return None, 0.0, annotated, [f"[...] Drawing... {len(self.trajectory_jz)} points"]
        
        # Show which finger to use:
        if self.target_letter == 'J':
            return None, 0.0, annotated, ["[>] Extend PINKY finger to start"]
        else:
            return None, 0.0, annotated, ["[>] Point INDEX finger UP to start"]
    
    def _classify_jz_trajectory(self):
        """Classify J or Z from trajectory with better distinction"""
        if len(self.trajectory_jz) < 15:
            return None
        
        traj = np.array(self.trajectory_jz)
        
        # Calculating overall motion characteristics:
        start = traj[0]
        end = traj[-1]
        
        # Total displacement:
        total_dx = end[0] - start[0]
        total_dy = end[1] - start[1]
        
        # Calculating direction changes:
        directions = []
        for i in range(1, len(traj)):
            dx = traj[i][0] - traj[i-1][0]
            dy = traj[i][1] - traj[i-1][1]
            if abs(dx) > 0.001 or abs(dy) > 0.001:  # Only if there's actual movement
                directions.append(np.arctan2(dy, dx))
        
        # Counting significant direction changes:
        changes = 0
        for i in range(1, len(directions)):
            diff = abs(directions[i] - directions[i-1])
            diff = min(diff, 2*np.pi - diff)
            if diff > np.pi/3:  # ~60 degree change
                changes += 1
        
        # Analyzing trajectory pattern:
        # J: Predominantly downward motion with ONE curve at the end
        # Z: Multiple direction changes with zigzag pattern
        
        # Checking if motion is predominantly downward (J characteristic):
        is_downward = total_dy > 0.1
        
        # Checking for horizontal motion (Z characteristic):
        has_horizontal = any(abs(traj[i+1][1] - traj[i][1]) < 0.02 and 
                           abs(traj[i+1][0] - traj[i][0]) > 0.03 
                           for i in range(len(traj)-1))
        
        # Decision logic:
        # When practicing a specific letter, be more lenient
        if self.target_letter == 'J':
            # Favor J unless very clearly Z (3+ changes with horizontal):
            if changes >= 3 or (changes >= 2 and has_horizontal):
                return 'Z'
            else:
                return 'J'
        elif self.target_letter == 'Z':
            # Favor Z unless very clearly J (0-1 changes, smooth):
            if changes <= 1 and is_downward and not has_horizontal:
                return 'J'
            else:
                return 'Z'
        else:
            # Free practice mode - standard logic:
            if changes >= 2 and has_horizontal:
                return 'Z'
            elif changes >= 2:
                return 'Z'
            elif changes == 1 and is_downward:
                return 'J'
            elif changes <= 1:
                return 'J'
            else:
                return self.target_letter if self.target_letter in ['J', 'Z'] else None
    
    def reset(self):
        """Reset recognition state"""
        self.prediction_history.clear()
        self.last_stable = None
        self.trajectory_jz = []


def enhance_frame(frame):
    """Enhance frame quality"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def draw_ui(frame, letter, confidence, system, feedback):
    """Draw comprehensive UI"""
    h, w = frame.shape[:2]
    
    # Main prediction:
    if letter and confidence > 0.3:
        is_correct = (letter == system.target_letter if system.practice_mode else False)
        color = (0, 255, 0) if is_correct else (255, 200, 0)
        
        cv2.rectangle(frame, (10, 10), (350, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 110), color, 4)
        cv2.putText(frame, str(letter), (30, 75),
                  cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)
        cv2.putText(frame, f"{confidence:.0%}", (30, 105),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Mode indicator:
    if system.practice_mode:
        cv2.rectangle(frame, (w-240, 10), (w-10, 75), (0, 0, 0), -1)
        cv2.rectangle(frame, (w-240, 10), (w-10, 75), (255, 100, 255), 3)
        cv2.putText(frame, "PRACTICE MODE", (w-230, 35),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
        cv2.putText(frame, f"Target: {system.target_letter}", (w-230, 65),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Feedback:
    if feedback:
        fb_y = 130
        for i, text in enumerate(feedback):
            color = (0, 255, 0) if '[OK]' in text else (100, 200, 255)
            cv2.putText(frame, text, (20, fb_y + i*30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Instructions:
    instr_y = h - 60
    cv2.rectangle(frame, (0, instr_y), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "a-z:Practice | +:Progress | SPACE:Reset | ESC:Quit/Exit", 
               (15, instr_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def main():
    model_path = project_root / "models" / "final_model_1.h5"
    class_indices_path = project_root / "models" / "class_indices_1.json"
    
    if not model_path.exists() or not class_indices_path.exists():
        print("Error: Model files not found")
        return
    
    print("="*70)
    print("COMPREHENSIVE ASL LEARNING SYSTEM - ALL 26 LETTERS")
    print("="*70)
    print("\nFeatures:")
    print("[+] All 26 letters including J and Z (motion-based)")
    print("[+] Visual finger position guide")
    print("[+] Real-time feedback")
    print("[+] Progress tracking")
    print("[+] Only recognizes LETTERS (no space/delete)")
    print("="*70)
    
    base_recognizer = ASLRecognizer(
        model_path=str(model_path),
        class_indices_path=str(class_indices_path)
    )
    
    system = ComprehensiveEducationalSystem(base_recognizer)
    
    print("\nStarting webcam...")
    print("\nControls:")
    print("a-z: Practice any letter (R works now!)")
    print("+: Show progress")
    print("SPACE: Reset")
    print("ESC: Quit or exit practice mode")
    print("\n" + "="*70 + "\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            processed = enhance_frame(frame)
            
            letter, confidence, annotated, feedback = system.process_frame(processed)
            
            # Adding finger guide if in practice mode and not doing J/Z:
            if system.practice_mode and system.target_letter not in ['J', 'Z']:
                results = system.recognizer.hand_detector.detect_hands(processed)
                if results.multi_hand_landmarks:
                    annotated = system.hand_analyzer.draw_finger_guide(
                        annotated, results.multi_hand_landmarks[0], system.target_letter)
            
            annotated = draw_ui(annotated, letter, confidence, system, feedback)
            
            cv2.imshow('ASL Learning - All 26 Letters', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                if system.practice_mode:
                    system.exit_practice()
                else:
                    break
            elif key == 32:  # SPACE bar for reset (frees up 'r')
                system.reset()
                print("Reset")
            elif key in [ord('='), ord('+')]:
                print("\n[PROGRESS REPORT]:")
                for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    stats = system.letter_stats[letter]
                    if stats['attempts'] > 0:
                        rate = (stats['successes']/stats['attempts'])*100
                        print(f"   {letter}: {rate:.0f}% ({stats['successes']}/{stats['attempts']})")
            elif 65 <= key <= 90 or 97 <= key <= 122:
                letter_key = chr(key).upper()
                system.start_practice(letter_key)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[DONE] Thanks for practicing!")


if __name__ == "__main__":
    main()
