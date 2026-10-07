import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time

# ================= CONFIG =================
BATCH_SIZE = 16
EPOCHS = 10
LR = 3e-5
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# ================= TRANSFORMS =================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ================= DATA =================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# ================= CLASS WEIGHTS =================
labels = [label for _, label in train_dataset]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# ================= MODEL =================
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes
)

model.to(DEVICE)

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ================= TRAINING =================
def train():
    print("\n===== Training Started =====")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate()

        print(f"Epoch {epoch+1} Completed")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Time: {time.time() - start_time:.2f} sec")

    print("\n===== Training Finished =====")


# ================= EVALUATION =================
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images).logits
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ================= MAIN =================
if __name__ == "__main__":
    train()