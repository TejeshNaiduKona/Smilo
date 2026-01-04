import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os

# Setup logs
#os.makedirs("logs", exist_ok=True)
#csv_path = "logs/Smilo_log.csv"
#if not os.path.exists(csv_path):
#    pd.DataFrame(columns=["timestamp", "emotion", "confidence"]).to_csv(csv_path, index=False)
# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to exit the webcam window...")

backends = [
    'opencv','ssd',                    # Fast and less Accurate
    'mtcnn','retinaface'               # Accurate and slow
]
detector = backends[2]
while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'],detector_backend = detector, enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][dominant_emotion]

        # Display emotion
        text = f"{dominant_emotion} ({confidence:.2f}%)"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Log to CSV
        #log = pd.DataFrame([[datetime.now(), dominant_emotion, confidence]],
        #                   columns=["timestamp", "emotion", "confidence"])
        #log.to_csv(csv_path, mode='a', header=False, index=False)

    except Exception as e:
        print("Error:", e)

    cv2.imshow('Smilo - Live Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
