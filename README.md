# Smilo - Real-Time Emotion Detection

A deep learning-based emotion detection system that uses Convolutional Neural Networks (CNN) to recognize facial expressions in real-time through a webcam feed.

## Features

- **Real-time emotion detection** using webcam
- **7 emotion classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **CNN architecture** with PyTorch for accurate predictions
- **Haar Cascade** face detection for efficient face localization
- **Optimized inference** with frame skipping for better performance

## Model Architecture

The FaceClassifier uses a 3-layer CNN architecture:
- 3 Convolutional layers (32, 64, 128 filters)
- MaxPooling after each convolution
- 2 Fully connected layers (512, num_classes)
- Dropout (0.3) for regularization
- Input size: 128x128 RGB images

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Smilo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 2.1.0
- torchvision 0.16.0
- OpenCV 4.8.1.78
- NumPy 1.24.3
- Pillow 10.0.0
- matplotlib 3.7.2

## Dataset Structure

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── validation/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Training

1. Prepare your dataset following the structure above
2. Open and run `Train_model.ipynb` notebook:
   - Configure CUDA settings
   - Load and transform datasets
   - Train the model (default: 20 epochs)
   - Save the trained model as `face_classifier.pth`
   - Save class labels as `classes.pkl`

The notebook includes:
- Data loading and preprocessing
- Model training with Adam optimizer
- Loss visualization
- Model evaluation on validation set

## Usage

Run the emotion detection system:

```bash
python main.py
```

**Controls:**
- Press `q` to quit the application

The system will:
1. Open your webcam
2. Detect faces using Haar Cascade
3. Predict emotions for detected faces
4. Display bounding boxes and emotion labels in real-time

## Project Files

- `main.py` - Real-time emotion detection application
- `model.py` - Model architecture and inference class
- `Train_model.ipynb` - Training notebook
- `face_classifier.pth` - Trained model weights
- `classes.pkl` - Emotion class labels
- `requirements.txt` - Python dependencies

## Performance Optimization

- Frame skipping (FRAME_SKIP=2) for faster processing
- GPU acceleration support (CUDA)
- Efficient face detection with largest face selection

## License

See the LICENSE file for details.

## Notes

- Ensure adequate lighting for better face detection
- Position your face clearly in front of the camera
- The model works best with frontal face views