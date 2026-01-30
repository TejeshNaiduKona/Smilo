import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchvision import transforms
from PIL import Image
import numpy as np


class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))



class EmotionPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open("classes.pkl", "rb") as f:
            self.classes = pickle.load(f)

        self.model = FaceClassifier(len(self.classes))
        self.model.load_state_dict(
            torch.load("face_classifier.pth", map_location=self.device)
        )
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    @torch.inference_mode()
    def predict(self, image_np: np.ndarray) -> str:
        img = Image.fromarray(image_np)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        return self.classes[output.argmax(1).item()]
