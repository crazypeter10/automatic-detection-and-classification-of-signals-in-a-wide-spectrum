import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import create_dataloader
from model import SignalCNN

# Setări pentru antrenament
data_dir = "data/train"
batch_size = 16
epochs = 100  # Creștem numărul de epoci
learning_rate = 0.0005  # Începem cu o rată de învățare ușor mai mare

# Inițializăm DataLoader-ul, modelul, funcția de pierdere și optimizatorul
train_loader = create_dataloader(data_dir, batch_size=batch_size)
model = SignalCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Adăugăm un scheduler pentru rata de învățare
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Funcția de antrenare
def train(model, train_loader, criterion, optimizer, scheduler, epochs=epochs):
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # Adăugăm pierderea la running_loss
            
        epoch_loss = running_loss / len(train_loader)  # Calculăm pierderea pe epocă
        scheduler.step(epoch_loss)  # Actualizăm rata de învățare pe baza pierderii epocii
        losses.append(epoch_loss)
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
    
    # Afișăm graficul pierderii (loss) pe epoci
    plt.plot(range(1, epochs+1), losses, marker='o')
    plt.xlabel("Epoca")
    plt.ylabel("Pierdere (Loss)")
    plt.title("Evoluția pierderii în timpul antrenamentului")
    plt.show()

# Executăm funcția de antrenare
if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, scheduler, epochs=epochs)
    torch.save(model.state_dict(), "model.pth")
    print("Modelul a fost salvat ca 'model.pth'")
