# Smilo 😃
**Real-Time Emotion Detection powered by PyTorch & OpenCV**

Smilo is a lightweight, deep learning-based application that detects and classifies 7 facial emotions (Angry 😠, Disgust 😐, Fear 😨, Happy 😃, Neutral 🙂, Sad 😔, Surprise 😮) in real-time. It features both a local desktop interface and a beautiful web-based interactive demo!

## 🚀 Getting Started

1. **Clone the repository** and navigate into the project directory:
   ```bash
   git clone <repository-url>
   cd Smilo
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Use

Smilo offers two distinct ways to interact with the emotion detection model:

### 1. Web Interface (Recommended)
Run a beautifully designed web app powered by Gradio. This interface supports uploading photos, capturing snapshots, or streaming live video directly from your webcam.
```bash
python app.py
```
*After running, click the local link (e.g. `http://127.0.0.1:XXXX`) in your terminal to open it in your browser.*

### 2. Desktop Application
Run the classic desktop script. This will instantly launch a video window using your webcam feed, drawing tracking boxes and emotion labels on detected faces.
```bash
python main.py
```
*Press `q` or click the 'X' button on the video window to quit.*

## 🧠 Model & Architecture

- **Face Detection:** Uses OpenCV's optimized Haar Cascades for rapid and highly-efficient face tracking. 
- **Emotion Recognition:** A custom 3-layer Convolutional Neural Network (CNN) built with PyTorch, trained on 128x128 resolution RGB images.
- **Performance:** System logic utilizes frame-skipping and concurrent processing optimizations to ensure video feeds maintain a lag-free 30+ FPS true 'live' experience.

## 🛠️ Retraining the Model

If you wish to augment the model or train it from scratch:
1. Ensure your dataset is prepared and sorted.
2. Open and run the `Train_model.ipynb` Jupyter Notebook.
3. The notebook will automatically guide you through data loading, transformation, model training, and exporting the updated inference weights (`face_classifier.pth`).

## Notes
Ensure you have adequate lighting and a clear, frontal view of your face for the most accurate predictions!

## License
See the LICENSE file for details.