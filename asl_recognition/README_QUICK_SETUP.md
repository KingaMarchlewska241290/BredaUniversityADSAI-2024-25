# ASL Learning System - Quick Setup Guide

## What This Does

Educational system for learning American Sign Language alphabet (all 26 letters). Uses MediaPipe hand detection and a trained model to provide real-time feedback while practicing.

## File Structure

Set up your project folder like this:

```
asl_recognition/
├── main.py                         # Main file
├── asl_recognizer.py               # Core recognition system
├── requirements.txt                # Python packages
│
├── utils/
│   ├── __init__.py
│   ├── hand_detector.py            # MediaPipe integration
│   └── dynamic_classifier.py       # Motion tracking for J/Z
│
└── models/
    ├── final_model_1.h5            # Trained model
    └── class_indices_1.json        # Letter mappings
```

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
cd asl_recognition
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python check_files.py        # Check all files are present
python check_environment.py  # Verify packages installed
```

### 3. Run It!

```bash
python main.py
```

## How to Use the Educational System

### FREE PRACTICE MODE (default)
Just make signs and see what the system recognizes.

### TARGETED PRACTICE MODE  
Press any letter key (a-z) to practice that specific letter:

1. **Press the letter key** (for example, press 'v' for letter V)
2. **Read instructions** shown in terminal
3. **Make the sign**
4. **Get feedback**:
   - If correct: "CORRECT!"
   - If wrong: Specific tips on what to fix
5. **Track progress** with '+' key

Example session:
```
Press 'v' --> Terminal shows: "Peace sign, 2 fingers APART"
          --> Make V sign
          --> Get: "CORRECT!" or "Show only 2 fingers (curl ring)"
Press '+' --> See: "V: 8/10 = 80% success rate"
```

## Controls

- **a-z** - Practice that letter (all 26 letters)
- **SPACE** - Reset predictions
- **+** - Show your progress stats
- **ESC** - Quit program or exit practice mode

## Key Features

### 1. All 26 Letters
- **Static (A-I, K-Y)**: Hold steady for 2-3 seconds
- **Dynamic (J, Z)**: Draw the shape in the air

### 2. Detailed Instructions
Each letter shows step-by-step instructions when you practice it.

### 3. Specific Feedback
System analyzes your hand and tells you exactly what to fix:
- "Move thumb to SIDE of fist" (A vs E)
- "Spread fingers APART more" (V vs K)  
- "Show 3 fingers (extend ring)" (W vs V)

### 4. Visual Finger Guide
Bottom of screen shows which fingers should be up/down.

### 5. Progress Tracking
See your success rate for each letter you've practiced.

### 6. Common Confusion Help
System knows which letters look similar and gives targeted advice.

## Tips for Best Results

1. **Good lighting** - Face a window or lamp
2. **Plain background** - Stand in front of a wall
3. **Center your hand** - Keep in middle of frame
4. **Hold steady** - Keep sign still for 2-3 seconds for static letters
5. **Clear signs** - Make exaggerated hand positions

## Practicing Dynamic Letters (J and Z)

J and Z require drawing motions in the air:

### Letter J:
1. Press 'j' to enter practice mode
2. Point index finger UP (others curled down)
3. Wait for "START DRAWING!" message
4. Draw J: Straight DOWN, then smooth hook LEFT
5. Keep motion flowing (like cursive j)
6. Purple line shows your path
7. System recognizes after 3 seconds

### Letter Z:
1. Press 'z' to enter practice mode
2. Point index finger UP
3. Wait for "START DRAWING!"
4. Draw Z: Diagonal down-right, horizontal left, diagonal down-right
5. Make sharp angles (clear zigzag pattern)
6. Include obvious horizontal middle segment
7. System recognizes after 3 seconds

**Tip**: When practicing J or Z, the system favors your target letter to help you learn.



## Required Files Checklist

Before running, make sure you have:

**Main Scripts:**
- [ ] main.py
- [ ] asl_recognizer.py
- [ ] requirements.txt

**Utils Folder:**
- [ ] utils/__init__.py
- [ ] utils/hand_detector.py
- [ ] utils/dynamic_classifier.py

**Models Folder:**
- [ ] models/final_model_1.h5
- [ ] models/class_indices_1.json

**Helper Scripts (optional but useful):**
- [ ] check_files.py
- [ ] check_environment.py



## Troubleshooting

### "ModuleNotFoundError: No module named 'utils'"
--> Make sure you're in the `asl_recognition` directory when running

### "Model file not found"
--> Ensure `models/final_model_1.h5` exists

### "Camera not accessible"
--> Check system permissions (Settings -> Privacy -> Camera)

### Predictions flickering/changing constantly?
--> System requires 60% consensus over 20 frames for stability

### Low accuracy for A, E, S?
--> These look similar. Use practice mode for specific feedback on thumb position and fist tightness.

### Low accuracy for V, W?
--> Use practice mode. System tells you exactly how many fingers to show.



## System Requirements

- Python 3.8+
- Webcam
- ~500MB RAM
- Works on: Mac, Windows, Linux



## Need Help?

Run diagnostics:
```bash
python check_environment.py
```

Check all files are present:
```bash
python check_files.py
```

---

**Ready to learn!** Run `python main_complete_fixed.py` and start practicing ASL :)
