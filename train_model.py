"""
PneumoScan - Proper Training Script
ResNet18 fine-tuned for pneumonia detection from chest X-rays.

Architecture: ResNet18 (pretrained ImageNet)
  - Unfrozen: layer3 + layer4 + FC
  - FC Head: Dropout(0.3) -> Linear(512, 2)
  - Loss: Weighted CrossEntropyLoss (handles 3:1 class imbalance)
  - Optimizer: Adam (lr=0.0001)
  - Scheduler: StepLR (step=5, gamma=0.5)
  - Epochs: 10

Target Metrics:
  - Train Accuracy: ~93%
  - Validation Accuracy: ~90%
  - Train Loss: ~0.20
  - Validation Loss: ~0.28
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import os
import time

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "data"
MODEL_SAVE_PATH = "model/pneumonia_model.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_WORKERS = 0  # Windows compatibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# DATA AUGMENTATION & TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# DATASETS & DATALOADERS
# ============================================================
print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "test"),
    transform=val_transform
)

print(f"Training classes: {train_dataset.class_to_idx}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ============================================================
# CLASS WEIGHTS (handle 3:1 imbalance)
# ============================================================
class_counts = [0, 0]
for _, label in train_dataset:
    class_counts[label] += 1
    if sum(class_counts) % 500 == 0:
        print(f"  Counting... {sum(class_counts)}/{len(train_dataset)}", end="\r")

total = sum(class_counts)
class_weights = [total / (2 * c) for c in class_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\nClass counts: NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}")
print(f"Class weights: NORMAL={class_weights[0]:.3f}, PNEUMONIA={class_weights[1]:.3f}")

# ============================================================
# MODEL SETUP
# ============================================================
print("\nBuilding model...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer3 + layer4 + FC (as per README)
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True

# Replace FC head with Dropout + Linear
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)

model = model.to(device)

# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

# ============================================================
# LOSS, OPTIMIZER, SCHEDULER
# ============================================================
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# ============================================================
# TRAINING LOOP
# ============================================================
print(f"\n{'='*60}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'='*60}\n")

best_val_acc = 0.0
os.makedirs("model", exist_ok=True)

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    train_loss /= train_total
    train_acc = train_correct / train_total

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    epoch_time = time.time() - epoch_start
    scheduler.step()

    # ---- LOGGING ----
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s)")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.1f}%")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

    # ---- SAVE BEST MODEL ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  >> Best model saved! (Val Acc: {val_acc*100:.1f}%)")

    print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"{'='*60}")
print(f"Training complete!")
print(f"Best Validation Accuracy: {best_val_acc*100:.1f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"{'='*60}")
