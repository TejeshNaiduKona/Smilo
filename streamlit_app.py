import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os
import numpy as np

st.set_page_config(page_title="Smilo", layout="wide")
st.title("🎭 Smilo - Real-Time Emotion, Gender & Age Detection")
st.markdown("Start the camera to detect your emotion, gender, and age in real time.")

# Checkbox to start/stop camera
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

# Create logs folder
#os.makedirs("logs", exist_ok=True)
#csv_path = "logs/Smilo_log.csv"
#if not os.path.exists(csv_path):
#    pd.DataFrame(columns=["timestamp", "emotion", "confidence", "gender", "age"]).to_csv(csv_path, index=False)

# Open webcam
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Cannot access camera")
        break

    try:
        # DeepFace analysis
        result = DeepFace.analyze(frame, actions=['emotion','age','gender'], enforce_detection=False)

        dominant_emotion = result[0]['dominant_emotion']
        emotion_confidence = result[0]['emotion'][dominant_emotion]

        predicted_gender = result[0]['gender']
        predicted_age = result[0]['age']
        age_min = int(predicted_age) - 3
        age_max = int(predicted_age) + 3
        age_range = f"{age_min}-{age_max}"

        # Overlay text
        text = f"{dominant_emotion} ({emotion_confidence:.0f}%), {predicted_gender}, Age: {age_range}"
        cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # Log CSV
        #log = pd.DataFrame([[datetime.now(), dominant_emotion, emotion_confidence, predicted_gender, predicted_age]],
        #                   columns=["timestamp","emotion","confidence","gender","age"])
        #log.to_csv(csv_path, mode='a', header=False, index=False)

    except Exception as e:
        st.warning(f"Detection Error: {e}")

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

camera.release()
