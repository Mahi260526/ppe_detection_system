
# PPE Detection System (Production Ready Template)

## Overview
This system detects people without proper PPE (helmet, vest, glasses) using YOLOv8.

## Setup Instructions

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Add Your Trained Model
Place your trained YOLOv8 PPE model as:
models/best.pt

### 3. Configure Video Source
Edit config.py:
- For webcam: VIDEO_SOURCE = 0
- For DroidCam:
  VIDEO_SOURCE = "http://<your-ip>:4747/video"

### 4. Run Application
python app.py

Press Q or Esc to quit.

### 5. View Violations Dashboard (optional)
Run the dashboard server in a separate terminal:
```
python server.py
```
Then open **http://127.0.0.1:5000** in a browser to see:
- KPIs: Total Violations, No Helmet, No Vest, No Glasses
- Recent violation images with date/time and tags

## Notes
- Train your PPE detection model using YOLOv8 before deployment.
- For production, integrate logging and alert system (email/SMS/API).
