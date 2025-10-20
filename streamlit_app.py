import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os
from streamlit.components.v1 import html

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Smilo 🎭 Live Emotion Detection", layout="wide")
st.markdown("""
<style>
/* Body background */
body {
    background-color: #f7f9fc;
}

/* Header card */
.header-card {
    background-color: #1f77b4;
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 2px 2px 10px #aaa;
}

/* Emotion cards */
.emotion-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 2px 2px 8px #ccc;
    text-align: center;
    margin: 10px;
}

/* Webcam frame */
.webcam-frame {
    border: 5px solid #1f77b4;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Styled Header
# -----------------------------
st.markdown("""
<div class='header-card'>
    <h1>🎭 Smilo - Real-Time Emotion Detection</h1>
    <p>Detect emotions live using AI + Webcam. Perfect for portfolio and demos.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Prepare logs
# -----------------------------
#os.makedirs("logs", exist_ok=True)
#csv_path = "logs/emotions_log.csv"
#if not os.path.exists(csv_path):
#    pd.DataFrame(columns=["timestamp", "emotion", "confidence"]).to_csv(csv_path, index=False)

# -----------------------------
# Start Camera Checkbox
# -----------------------------
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.empty()  # Placeholder for webcam frame

# Columns for Emotion Cards
col1, col2, col3 = st.columns(3)

# Initialize webcam
camera = cv2.VideoCapture(0)

# Emotion counters for cards
emotion_counts = {"happy":0, "sad":0, "angry":0, "surprise":0, "neutral":0, "fear":0, "disgust":0}

# -----------------------------
# Webcam Loop
# -----------------------------
while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not detected.")
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion'].lower()
        confidence = result[0]['emotion'][dominant_emotion]

        # Increment emotion count
        if dominant_emotion in emotion_counts:
            emotion_counts[dominant_emotion] += 1

        # Overlay emotion on frame
        cv2.putText(frame, f"{dominant_emotion.capitalize()} ({confidence:.2f}%)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Log to CSV
        #log = pd.DataFrame([[datetime.now(), dominant_emotion, confidence]],
        #                   columns=["timestamp", "emotion", "confidence"])
        #log.to_csv(csv_path, mode='a', header=False, index=False)

    except:
        pass

    # Display webcam frame with border
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

    # -----------------------------
    # Display Emotion Cards
    # -----------------------------
    for col, emotion in zip([col1, col2, col3], list(emotion_counts.keys())[:3]):
        col.markdown(f"""
        <div class='emotion-card'>
            <h3>{emotion.capitalize()}</h3>
            <h2>{emotion_counts[emotion]}</h2>
        </div>
        """, unsafe_allow_html=True)
