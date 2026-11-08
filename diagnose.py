"""
Diagnostic script to check why the model always predicts Pneumonia.
Tests on actual images from both classes and prints raw logics.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import Counter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---- Load model exactly like app.py does ----
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)
model.load_state_dict(torch.load("model/pneumonia_model.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

# ---- Same transform as app.py ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Check class ordering ----
print("\n--- Class Ordering Check ---")
test_dataset = datasets.ImageFolder(root="data/test", transform=transform)
print(f"ImageFolder class_to_idx: {test_dataset.class_to_idx}")
print(f"app.py classes list:      ['Normal', 'Pneumonia']")
print(f"Match: NORMAL->0, PNEUMONIA->1? {test_dataset.class_to_idx.get('NORMAL') == 0}")

# ---- Test on test set ----
print("\n--- Testing on FULL test set ---")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_labels = []
all_logits = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_logits.extend(outputs.cpu().numpy())

pred_counts = Counter(all_preds)
label_counts = Counter(all_labels)

print(f"True labels distribution:  {{0 (Normal): {label_counts[0]}, 1 (Pneumonia): {label_counts[1]}}}")
print(f"Predicted distribution:    {{0 (Normal): {pred_counts[0]}, 1 (Pneumonia): {pred_counts[1]}}}")

# Show accuracy per class
normal_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 0 and p == 0)
pneumonia_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == 1 and p == 1)
normal_total = label_counts[0]
pneumonia_total = label_counts[1]

print(f"\nNormal accuracy:    {normal_correct}/{normal_total} = {normal_correct/normal_total*100:.1f}%")
print(f"Pneumonia accuracy: {pneumonia_correct}/{pneumonia_total} = {pneumonia_correct/pneumonia_total*100:.1f}%")
print(f"Overall accuracy:   {(normal_correct+pneumonia_correct)/(normal_total+pneumonia_total)*100:.1f}%")

# Show a few raw logits
import numpy as np
logits = np.array(all_logits)
labels = np.array(all_labels)

print("\n--- Sample Raw Logits (first 5 Normal images) ---")
normal_indices = np.where(labels == 0)[0][:5]
for i in normal_indices:
    prob = torch.softmax(torch.tensor(logits[i]), dim=0).numpy()
    print(f"  Logits: [{logits[i][0]:.4f}, {logits[i][1]:.4f}]  "
          f"Probs: [Normal={prob[0]:.4f}, Pneumonia={prob[1]:.4f}]  "
          f"Pred: {'Normal' if all_preds[i]==0 else 'PNEUMONIA'}")

print("\n--- Sample Raw Logits (first 5 Pneumonia images) ---")
pneumonia_indices = np.where(labels == 1)[0][:5]
for i in pneumonia_indices:
    prob = torch.softmax(torch.tensor(logits[i]), dim=0).numpy()
    print(f"  Logits: [{logits[i][0]:.4f}, {logits[i][1]:.4f}]  "
          f"Probs: [Normal={prob[0]:.4f}, Pneumonia={prob[1]:.4f}]  "
          f"Pred: {'Normal' if all_preds[i]==0 else 'PNEUMONIA'}")
