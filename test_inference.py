"""
Simulate app.py inference on specific test images to verify the pipeline.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model exactly like app.py
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load("model/pneumonia_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Same transform as app.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["Normal", "Pneumonia"]

# Test known Normal images (exactly like app.py does)
print("=== Testing NORMAL images (should predict Normal) ===")
normal_dir = "data/test/NORMAL"
for fname in os.listdir(normal_dir)[:5]:
    fpath = os.path.join(normal_dir, fname)
    image = Image.open(fpath).convert("RGB")  # same as app.py
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
    print(f"  {fname}: {classes[pred.item()]} (Normal={probs[0][0]:.3f}, Pneumonia={probs[0][1]:.3f})")

# Test known Pneumonia images
print("\n=== Testing PNEUMONIA images (should predict Pneumonia) ===")
pneumonia_dir = "data/test/PNEUMONIA"
for fname in os.listdir(pneumonia_dir)[:5]:
    fpath = os.path.join(pneumonia_dir, fname)
    image = Image.open(fpath).convert("RGB")
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
    print(f"  {fname}: {classes[pred.item()]} (Normal={probs[0][0]:.3f}, Pneumonia={probs[0][1]:.3f})")
