import torch
import torch.optim as optim
import torch.nn as nn
from dataset import create_dataloader
from model import SignalCNN
import matplotlib.pyplot as plt

# Setări pentru antrenament
data_dir = "data/train"
batch_size = 16
epochs = 50  # Creștem numărul de epoci
learning_rate = 0.0001  # Reducem rata de învățare

# Inițializăm DataLoader-ul, modelul, funcția de pierdere și optimizatorul
train_loader = create_dataloader(data_dir, batch_size=batch_size)
model = SignalCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Funcția de antrenare
def train(model, train_loader, criterion, optimizer, epochs=epochs):
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
    
    # Afișăm graficul pierderii (loss) pe epoci
    plt.plot(range(1, epochs+1), losses, marker='o')
    plt.xlabel("Epoca")
    plt.ylabel("Pierdere (Loss)")
    plt.title("Evoluția pierderii în timpul antrenamentului")
    plt.show()

# Executăm funcția de antrenare
if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, epochs=epochs)
    
    # Salvăm modelul antrenat
    torch.save(model.state_dict(), "model.pth")
    print("Modelul a fost salvat ca 'model.pth'")
