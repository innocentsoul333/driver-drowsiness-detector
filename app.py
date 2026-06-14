import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
import os
import urllib.request
import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av

try:
    import winsound
except ImportError:
    class winsound:
        SND_ALIAS = 0
        SND_ASYNC = 0
        @staticmethod
        def PlaySound(sound, flags):
            pass

# --- Configuration ---
EAR_THRESHOLD = 0.25 
DROWSINESS_TIME_THRESHOLD = 1.5
MAR_THRESHOLD = 0.45             
YAWN_TIME_THRESHOLD = 1.0        
MODEL_PATH = "face_landmarker.task"

# Landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- Helper Functions ---
@st.cache_resource(show_spinner="Downloading Model...")
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
    return True

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

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Neuromorphic Drowsiness Detection", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Neuromorphic UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        text-align: center;
        color: #38bdf8;
        font-weight: 800;
        letter-spacing: 3px;
        margin-top: -50px;
        margin-bottom: 30px;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        text-transform: uppercase;
    }
    .neuromorphic-panel {
        background: #0f172a;
        border-radius: 20px;
        box-shadow: 8px 8px 16px #080c17, -8px -8px 16px #16223d;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 15px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 1rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        font-weight: 600;
    }
    .log-box {
        height: 380px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: #a7f3d0;
        padding: 10px;
    }
    .log-entry {
        padding: 10px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        align-items: center;
    }
    .log-entry::before {
        content: "►";
        color: #38bdf8;
        margin-right: 12px;
        font-size: 0.8rem;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a; 
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155; 
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.6rem;
            margin-top: -20px;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 2.5rem;
            margin: 10px 0;
        }
        .neuromorphic-panel {
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 12px;
        }
        .log-box {
            height: 200px;
            font-size: 0.85rem;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>NEUROMORPHIC EVENT MONITOR</h1>", unsafe_allow_html=True)

# WebRTC setup
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        download_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.eyes_closed_start_time = None
        self.yawn_start_time = None
        self.eye_spike_fired = False
        self.yawn_spike_fired = False
        
        self.current_status = "ACTIVE"
        self.status_color = "#38bdf8"
        self.neural_spikes = 0
        self.event_log = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)
        
        current_status = "ACTIVE"
        status_color = "#38bdf8"
        avg_ear = 0
        mar = 0
        
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                left_eye_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in LEFT_EYE_INDICES]
                right_eye_points = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in RIGHT_EYE_INDICES]

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(face_landmarks, w, h)
                
                current_time_str = datetime.datetime.now().strftime("%H:%M:%S")

                # Eye Logic
                if avg_ear < EAR_THRESHOLD:
                    if self.eyes_closed_start_time is None:
                        self.eyes_closed_start_time = time.time()
                    else:
                        if (time.time() - self.eyes_closed_start_time) > DROWSINESS_TIME_THRESHOLD:
                            current_status = "DROWSY"
                            status_color = "#ef4444"
                            if not self.eye_spike_fired:
                                self.neural_spikes += 1
                                self.event_log.append(f"[{current_time_str}] Eye Closure Spike")
                                self.eye_spike_fired = True
                                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
                else:
                    self.eyes_closed_start_time = None
                    self.eye_spike_fired = False

                # Yawn Logic
                if mar > MAR_THRESHOLD:
                    if self.yawn_start_time is None:
                        self.yawn_start_time = time.time()
                    else:
                        if (time.time() - self.yawn_start_time) > YAWN_TIME_THRESHOLD:
                            if current_status != "DROWSY":
                                current_status = "YAWNING"
                                status_color = "#f97316"
                            if not self.yawn_spike_fired:
                                self.neural_spikes += 1
                                self.event_log.append(f"[{current_time_str}] Yawn Spike")
                                self.yawn_spike_fired = True
                                winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                else:
                    self.yawn_start_time = None
                    self.yawn_spike_fired = False

                # Draw minimal mesh
                for pt in left_eye_points + right_eye_points:
                    cv2.circle(img, pt, 2, (0, 255, 255), -1)
                
                mouth_pts = [13, 14, 61, 291]
                for pt_idx in mouth_pts:
                    pt = (int(face_landmarks[pt_idx].x * w), int(face_landmarks[pt_idx].y * h))
                    cv2.circle(img, pt, 3, (255, 0, 255), -1)
                    
                # HUD Text
                cv2.putText(img, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f}", (15, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            
        self.current_status = current_status
        self.status_color = status_color
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def render_ui_state(status, status_color, spikes, logs):
    status_html = f"""
    <div class='neuromorphic-panel'>
        <div class='metric-label'>System Status</div>
        <div class='metric-value' style='color: {status_color}; text-shadow: 0 0 20px {status_color}80;'>{status}</div>
    </div>
    """
    status_placeholder.markdown(status_html, unsafe_allow_html=True)
    
    spikes_color = "#fde047" if spikes > 0 else "#4ade80"
    spikes_html = f"""
    <div class='neuromorphic-panel'>
        <div class='metric-label'>Neural Spikes</div>
        <div class='metric-value' style='color: {spikes_color}; text-shadow: 0 0 20px {spikes_color}80;'>{spikes}</div>
    </div>
    """
    spikes_placeholder.markdown(spikes_html, unsafe_allow_html=True)
    
    log_content = "".join([f"<div class='log-entry'>{entry}</div>" for entry in reversed(logs[-15:])])
    if not log_content:
        log_content = "<div class='log-entry' style='color: #475569; border: none;'>No events logged...</div>"
    
    log_html = f"<div class='log-box'>{log_content}</div>"
    log_placeholder.markdown(log_html, unsafe_allow_html=True)

# Layout
col1, col_gap, col2 = st.columns([2.2, 0.1, 1])

with col1:
    st.markdown("<div class='neuromorphic-panel' style='padding: 15px;'>", unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="drowsiness",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=DrowsinessProcessor,
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    status_placeholder = st.empty()
    spikes_placeholder = st.empty()
    
    st.markdown("""
    <div class='neuromorphic-panel' style='padding: 15px;'>
        <div class='metric-label' style='margin-bottom: 15px;'>Event Log</div>
    """, unsafe_allow_html=True)
    log_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

if webrtc_ctx.state.playing:
    while True:
        if webrtc_ctx.video_processor:
            p = webrtc_ctx.video_processor
            render_ui_state(p.current_status, p.status_color, p.neural_spikes, p.event_log)
        time.sleep(0.5)
else:
    render_ui_state("STANDBY", "#94a3b8", 0, [])
