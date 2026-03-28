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
        return None, "Wait! No image provided. Please upload or capture an image."
    
    # image is a numpy array (RGB) from Gradio type="numpy"
    frame = image
    
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
        return frame, "No faces detected in the image 👀"
    
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
        
        # Draw rectangle and label with a slightly thicker and smoother style
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 105, 180), 3) # Hot pink bounding box
        
        # Text background for better legibility
        (text_width, text_height), _ = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.rectangle(output_frame, (x, y - text_height - 15), (x + text_width + 10, y), (255, 105, 180), -1)
        # Put white text over the pink box
        cv2.putText(
            output_frame, emotion, (x + 5, y - 10),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2
        )
    
    # Convert back to RGB for display
    output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    
    # Return annotated image and detected emotion
    emotion_text = ", ".join(emotions) if emotions else "Unknown"
    
    return output_frame_rgb, f"Detected Emotion: {emotion_text} ✨"


# Custom CSS for a modern, sleek UI
custom_css = """
body {
    background-color: #f8fafc;
}
.gradio-container {
    font-family: 'Inter', sans-serif !important;
}
.header-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    padding: 2.5rem;
    border-radius: 1rem;
    color: white;
    text-align: center;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}
.header-card h1 {
    color: white !important;
    font-size: 3rem !important;
    margin-bottom: 0.5rem !important;
    font-weight: 800;
}
.header-card p {
    font-size: 1.25rem;
    opacity: 0.9;
    font-weight: 400;
}
.zone-card {
    background: white;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border: 1px solid #e2e8f0;
}
.detect-btn {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    transition: all 0.2s ease-in-out !important;
}
.detect-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4) !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
)

with gr.Blocks(theme=theme, css=custom_css, title="Smilo😃 - Real-Time Emotion Detection") as demo:
    
    with gr.Row():
        gr.HTML('''
            <div class="header-card">
                <h1>Smilo 😃</h1>
                <p>Real-Time Emotion Detection powered by PyTorch</p>
            </div>
        ''')
        
    with gr.Row():
        with gr.Column(elem_classes=["zone-card"]):
            gr.Markdown("### 📸 Input Frame\nUpload an image or use your webcam to analyze facial expressions.")
            image_input = gr.Image(
                label="Image Source",
                type="numpy",
                sources=["upload", "webcam"],
                elem_id="image-input"
            )
            submit_btn = gr.Button("✨ Predict Emotion", elem_classes=["detect-btn"])
            
        with gr.Column(elem_classes=["zone-card"]):
            gr.Markdown("### 🎯 Analysis Results\nHere is what Smilo discovered in the image!")
            image_output = gr.Image(label="Annotated Image", type="numpy")
            emotion_output = gr.Textbox(
                label="Prediction Result", 
                interactive=False, 
                lines=2
            )
            
    with gr.Accordion("ℹ️ About Smilo", open=False):
        gr.Markdown("""
        **Smilo😃** uses a custom-trained Convolutional Neural Network (CNN) in PyTorch to detect facial emotions. 
        It first leverages an OpenCV Haar Cascade face detector to locate faces in the image. The cropped facial regions are then evaluated by our CNN to determine the dominant emotion.
        """)

    # Event handlers
    # Button click
    submit_btn.click(
        fn=predict_emotion,
        inputs=[image_input],
        outputs=[image_output, emotion_output]
    )
    # Automatic prediction on change
    image_input.change(
        fn=predict_emotion,
        inputs=[image_input],
        outputs=[image_output, emotion_output]
    )


if __name__ == "__main__":
    demo.launch()
