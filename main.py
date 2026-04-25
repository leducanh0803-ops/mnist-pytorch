# ============================================================
#  MNIST in PyTorch — Annotated Walkthrough
#  Run section by section to build intuition
# ============================================================

# ── 0. Imports ───────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    # ── 1. DATA ──────────────────────────────────────────────────
    # transforms.Compose chains multiple transforms together
    # ToTensor()     : PIL Image (H x W x C, uint8) → Tensor (C x H x W, float32 in [0,1])
    # Normalize(...) : (pixel - mean) / std  →  roughly in [-1, 1]
    #                  values are (mean,) and (std,) per channel; MNIST is grayscale → 1 channel
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST dataset mean & std
    ])

    # torchvision downloads the dataset for you on first run
    train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # DataLoader wraps a Dataset and handles batching + shuffling
    # num_workers > 0 loads data in parallel (helpful for large datasets)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2)

    # ✅ Sanity check — always do this before building your model
    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")   # → torch.Size([64, 1, 28, 28])
    print(f"Labels shape: {labels.shape}")   # → torch.Size([64])
    # 64 images, 1 channel, 28x28 pixels


    # ── 2. MODEL ─────────────────────────────────────────────────
    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()

            # nn.Sequential is a convenient container — layers run in order
            self.network = nn.Sequential(
                nn.Flatten(),           # (batch, 1, 28, 28) → (batch, 784)
                nn.Linear(784, 128),    # fully connected: 784 inputs → 128 neurons
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),      # 10 outputs = 10 digit classes
                # ⚠️  NO softmax here — CrossEntropyLoss applies it internally (log-softmax + NLLLoss)
                #     Adding softmax yourself is a common mistake from Keras habits
            )

        def forward(self, x):
            return self.network(x)      # just call the sequential block


    # ── 3. SETUP ─────────────────────────────────────────────────
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model     = MNISTNet().to(device)          # move model weights to device
    criterion = nn.CrossEntropyLoss()          # loss function (expects raw logits)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ✅ Sanity check — print model structure
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")   # → 109,386


    # ── 4. TRAINING LOOP ─────────────────────────────────────────
    def train_one_epoch(model, loader, criterion, optimizer, device):
        model.train()       # sets training mode (enables dropout, batchnorm, etc.)
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)  # move data to device

            # ── The 5-step PyTorch training rhythm ──
            optimizer.zero_grad()               # 1. clear gradients from last step
            outputs = model(images)             # 2. forward pass → raw logits [batch, 10]
            loss    = criterion(outputs, labels)# 3. compute loss
            loss.backward()                     # 4. backprop — compute gradients
            optimizer.step()                    # 5. update weights

            total_loss += loss.item()           # .item() extracts the scalar value from tensor

        avg_loss = total_loss / len(loader)
        return avg_loss


    # ── 5. EVALUATION LOOP ───────────────────────────────────────
    def evaluate(model, loader, criterion, device):
        model.eval()        # disables dropout, batchnorm uses running stats
        total_loss    = 0
        correct       = 0

        with torch.no_grad():   # don't track gradients — saves memory & speeds things up
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                total_loss += loss.item()
                preds   = outputs.argmax(dim=1)     # pick class with highest logit
                correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = correct / len(loader.dataset)
        return avg_loss, accuracy


    # ── 6. TRAINING RUN ──────────────────────────────────────────
    NUM_EPOCHS = 5

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss            = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc   = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}/{NUM_EPOCHS}  "
              f"train_loss: {train_loss:.4f}  "
              f"test_loss: {test_loss:.4f}  "
              f"test_acc: {test_acc*100:.2f}%")

    # Expected output after 5 epochs:
    # Epoch 1/5  train_loss: 0.2731  test_loss: 0.1162  test_acc: 96.52%
    # Epoch 5/5  train_loss: 0.0451  test_loss: 0.0812  test_acc: 97.60%


    # ── 7. SAVE & LOAD (bonus) ───────────────────────────────────
    # Save only the weights (recommended)
    torch.save(model.state_dict(), "mnist_model.pth")

    # Load back
    model2 = MNISTNet().to(device)
    model2.load_state_dict(torch.load("mnist_model.pth"))
    model2.eval()
    print("Model loaded successfully ✅")

if __name__ == '__main__':
    main()