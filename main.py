import cv2
from model import EmotionPredictor

cap = cv2.VideoCapture(0)
predictor = EmotionPredictor()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade")

FRAME_SKIP = 2
frame_count = 0
faces = []


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 301)
        )
        faces = [max(detected, key=lambda r: r[2]*r[3])] if len(detected) else []

    for (x, y, w, h) in faces:
        face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        label = predictor.predict(face_rgb)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


