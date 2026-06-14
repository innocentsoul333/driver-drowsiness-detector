# Neuromorphic Driver Drowsiness Detection

A beginner-friendly Python application featuring a futuristic "Neuromorphic Event Monitor" that detects driver drowsiness using your computer's webcam. It leverages OpenCV and MediaPipe Face Mesh to calculate both Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) in real-time.

This repository includes **two versions**:
1. A modern **Streamlit Web Application** (`app.py`)
2. A standalone **OpenCV Window Application** (`main.py`)

## Features
- **Neuromorphic Event Panel:** A futuristic web HUD that displays live event tracking and real-time logs.
- **Eye Closure Detection:** Calculates Eye Aspect Ratio (EAR) to determine if your eyes have been closed beyond the safety threshold.
- **Yawn Detection:** Calculates Mouth Aspect Ratio (MAR) to monitor yawning events.
- **Neural Spikes:** Instead of spamming continuous alarms, the system intelligently logs singular "spikes" in a chronological Event Log whenever an event happens.

## Prerequisites
- Python 3.9+
- A connected webcam

## Setup Instructions

1. **Clone this repository** (or download as ZIP):
   ```bash
   git clone https://github.com/chinmaymohite3036/driver-drowsiness-detector.git
   cd driver-drowsiness-detector
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   
   # Activate on Windows:
   venv\Scripts\activate
   
   # Activate on Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web Application (Recommended)

To launch the beautiful, modern browser-based Neuromorphic HUD:

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
2. Streamlit will open a web page in your default browser (usually at `http://localhost:8501`).
3. Click the **START** button on the video panel. Your browser will ask for camera permission—click **Allow**.
4. Test it by closing your eyes for 1.5 seconds or opening your mouth wide (yawning) for 1 second!

## Running the Desktop Application

If you prefer a standalone local window without a web browser (uses the native `winsound` alerts on Windows):

1. **Run the main script**:
   ```bash
   python main.py
   ```
2. A window will open showing your live webcam feed. 
3. **Press `q`** while the webcam window is active to exit.
