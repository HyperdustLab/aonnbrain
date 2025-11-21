#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

class MnistPipeline(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.pipeline(x)
        logits = self.classifier(h)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MnistPipeline().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loader.__dict__
def train_epoch(epoch):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
    avg_loss = total_loss / total
    acc = total_correct / total * 100
    print(f"Epoch {epoch:02d}: loss={avg_loss:.4f}, acc={acc:.2f}%")

def evaluate():
    model.eval()
    total_correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    print(f"Test accuracy: {total_correct / total * 100:.2f}%")

if __name__ == "__main__":
    for epoch in range(1, 6):
        train_epoch(epoch)
        evaluate()
