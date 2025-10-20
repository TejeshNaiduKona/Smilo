import cv2
from deepface import DeepFace
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Crop the face for deepface
                # face_crop = frame[y:y+h, x:x+w]
                pad = 20
                y1 = max(0, y - pad)
                y2 = min(frame.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(frame.shape[1], x + w + pad)
                face_crop = frame[y1:y2, x1:x2]


                try:
                    analysis = DeepFace.analyze(
                        img_path=face_crop,
                        actions=['emotion', 'age', 'gender'],
                        enforce_detection=False,
                        # prog_bar=False,
                        # detector_backend='opencv',
                        detector_backend='mtcnn'
                    )


                    emotion = analysis[0]['dominant_emotion']
                    age = analysis[0]['age']
                    gender = analysis[0]['gender']


                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{gender}, {age}, {emotion}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                except Exception as e:
                    print("DeepFace error:", e)


        cv2.imshow('Real-Time Emotion Detection', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()