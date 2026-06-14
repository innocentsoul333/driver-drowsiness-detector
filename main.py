import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import winsound  # Native Windows module for sound
import math
import os
import urllib.request
import datetime

# --- Configuration ---
EAR_THRESHOLD = 0.25 
DROWSINESS_TIME_THRESHOLD = 1.5  # Time in seconds eyes must be closed
MAR_THRESHOLD = 0.45             # Mouth Aspect Ratio for yawn
YAWN_TIME_THRESHOLD = 1.0        # Time in seconds mouth must be open to be a yawn
MODEL_PATH = "face_landmarker.task"

# Landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Lips: 13 (inner top), 14 (inner bottom), 61 (outer left corner), 291 (outer right corner)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH} (this might take a moment)...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Download complete!")

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(eye_landmarks):
    v1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    h = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

def calculate_mar(face_landmarks, w, h):
    pt13 = (face_landmarks[13].x * w, face_landmarks[13].y * h)
    pt14 = (face_landmarks[14].x * w, face_landmarks[14].y * h)
    pt61 = (face_landmarks[61].x * w, face_landmarks[61].y * h)
    pt291 = (face_landmarks[291].x * w, face_landmarks[291].y * h)
    
    vertical = euclidean_distance(pt13, pt14)
    horizontal = euclidean_distance(pt61, pt291)
    
    return vertical / horizontal if horizontal != 0 else 0

def draw_futuristic_ui(frame, w, h, spikes, event_log):
    # Create an overlay for the Neuromorphic Panel
    panel_width = 350
    overlay = frame.copy()
    
    # Dark panel background on the right side
    cv2.rectangle(overlay, (w - panel_width, 0), (w, h), (20, 15, 25), -1)
    # Apply transparency
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Neon Cyan border
    cv2.line(frame, (w - panel_width, 0), (w - panel_width, h), (255, 255, 0), 2)
    
    # UI Text Origin
    x_start = w - panel_width + 20
    
    # Brain/Neural Header
    cv2.putText(frame, "NEUROMORPHIC", (x_start, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 100, 0), 2)
    cv2.putText(frame, "EVENT MONITOR", (x_start, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 100, 0), 2)
    cv2.line(frame, (x_start, 85), (w - 20, 85), (255, 255, 0), 1)
    
    # Neural Spikes Counter
    cv2.putText(frame, "NEURAL SPIKES:", (x_start, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Futuristic spike number display
    spike_color = (0, 255, 255) if spikes > 0 else (0, 255, 0) # Yellow if spikes exist, green if 0
    cv2.putText(frame, f"{spikes}", (x_start + 180, 135), cv2.FONT_HERSHEY_DUPLEX, 1.2, spike_color, 2)
    
    # Graph/Visualizer simulation (Static for now)
    cv2.rectangle(frame, (x_start, 160), (w - 20, 200), (40, 40, 50), -1)
    for i in range(10):
        intensity = np.random.randint(5, 35) if spikes > 0 else 5
        cv2.line(frame, (x_start + 5 + i*30, 195), (x_start + 5 + i*30, 195 - intensity), spike_color, 4)
        
    cv2.line(frame, (x_start, 220), (w - 20, 220), (255, 255, 0), 1)

    # Event Log Section
    cv2.putText(frame, "EVENT LOG", (x_start, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    
    # Display recent events (up to 10)
    log_y = 280
    for event in event_log[-10:]:
        cv2.putText(frame, event, (x_start, log_y), cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 255, 200), 1)
        log_y += 25

def main():
    download_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Expand camera width to make room for the UI if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting Neuromorphic Event Monitor...")
    
    eyes_closed_start_time = None
    yawn_start_time = None
    
    eye_spike_fired = False
    yawn_spike_fired = False
    
    neural_spike_counter = 0
    event_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # Eye calculation
                left_eye_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_EYE_INDICES]
                right_eye_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_EYE_INDICES]

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Mouth calculation
                mar = calculate_mar(face_landmarks, w, h)
                
                current_time_str = datetime.datetime.now().strftime("%H:%M:%S")

                # Eye Spike Logic
                if avg_ear < EAR_THRESHOLD:
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                    else:
                        if (time.time() - eyes_closed_start_time) > DROWSINESS_TIME_THRESHOLD:
                            if not eye_spike_fired:
                                neural_spike_counter += 1
                                event_log.append(f"[{current_time_str}] Eye Closure Spike")
                                eye_spike_fired = True
                                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
                else:
                    eyes_closed_start_time = None
                    eye_spike_fired = False

                # Yawn Spike Logic
                if mar > MAR_THRESHOLD:
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    else:
                        if (time.time() - yawn_start_time) > YAWN_TIME_THRESHOLD:
                            if not yawn_spike_fired:
                                neural_spike_counter += 1
                                event_log.append(f"[{current_time_str}] Yawn Spike")
                                yawn_spike_fired = True
                                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                else:
                    yawn_start_time = None
                    yawn_spike_fired = False

                # Draw Mesh (minimalistic futuristic mesh mapping)
                for pt in left_eye_points + right_eye_points:
                    cv2.circle(frame, pt, 1, (0, 255, 255), -1)
                
                mouth_pts = [13, 14, 61, 291]
                for pt_idx in mouth_pts:
                    pt = (int(face_landmarks[pt_idx].x * w), int(face_landmarks[pt_idx].y * h))
                    cv2.circle(frame, pt, 2, (255, 0, 255), -1)

                cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)

        # Draw UI Panel
        draw_futuristic_ui(frame, w, h, neural_spike_counter, event_log)

        cv2.imshow("Neuromorphic Event Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
