import gradio as gr
import cv2
import numpy as np
from model import EmotionPredictor


# Initialize the predictor
predictor = EmotionPredictor()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade")


def predict_emotion(image):
    """
    Predict emotion from an image.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        annotated image and emotion prediction
    """
    if image is None:
        return None, "No image provided"
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, np.ndarray):
        frame = image
    else:
        frame = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    detected = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    if len(detected) == 0:
        return frame, "No face detected"
    
    # Get the largest face
    faces = [max(detected, key=lambda r: r[2]*r[3])]
    
    # Process the face
    output_frame = frame_bgr.copy()
    emotions = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_rgb = cv2.cvtColor(frame_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        
        # Predict emotion
        emotion = predictor.predict(face_rgb)
        emotions.append(emotion)
        
        # Draw rectangle and label
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            output_frame, emotion, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
    
    # Convert back to RGB for display
    output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    
    # Return annotated image and detected emotion
    emotion_text = ", ".join(emotions) if emotions else "No emotion detected"
    
    return output_frame_rgb, f"Detected emotion(s): {emotion_text}"


# Create Gradio interface
with gr.Blocks(title="# Smilo😃 - Real-Time Emotion Detection") as demo:
    gr.Markdown("# Smilo😃 - Real-Time Emotion Detection")
    gr.Markdown("Upload an image or use your webcam to detect facial emotions")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Input Image",
                type="pil",
                sources=["upload", "webcam"]
            )
            submit_btn = gr.Button("Predict Emotion", variant="primary")
        
        with gr.Column():
            image_output = gr.Image(label="Annotated Image")
            emotion_output = gr.Textbox(label="Prediction Result", interactive=False)
    
    # Connect the function to the button
    submit_btn.click(
        fn=predict_emotion,
        inputs=[image_input],
        outputs=[image_output, emotion_output]
    )
    
    # Also run prediction when image is uploaded
    image_input.change(
        fn=predict_emotion,
        inputs=[image_input],
        outputs=[image_output, emotion_output]
    )


if __name__ == "__main__":
    demo.launch(share=True)
