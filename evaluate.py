import torch
from dataset import create_dataloader
from model import SignalCNN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Configurare pentru evaluare
data_dir = "data/test"  # Asigură-te că ai datele de test în această locație
batch_size = 16
classes = ["sinusoidal", "noisy", "interference"]

# Încarcă setul de date de testare și modelul
test_loader = create_dataloader(data_dir, batch_size=batch_size)
model = SignalCNN()
model.load_state_dict(torch.load("model.pth", weights_only=True))  # Încărcăm modelul salvat
model.eval()  # Setăm modelul în modul de evaluare

# Funcția de evaluare și afișare a acurateței
def evaluate(model, test_loader):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():  # Dezactivăm calculul gradientului pentru evaluare
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / total
    print(f'Acuratețea modelului pe setul de testare este: {accuracy}%')
    
    # Afișăm matricea de confuzie
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("Matricea de Confuzie")
    plt.show()

# Afișăm exemple corecte și incorecte
def display_classification_examples(model, test_loader, classes):
    model.eval()
    correct = []
    incorrect = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                if predicted[i] == labels[i] and len(correct) < 5:
                    correct.append((inputs[i], predicted[i], labels[i]))
                elif predicted[i] != labels[i] and len(incorrect) < 5:
                    incorrect.append((inputs[i], predicted[i], labels[i]))
                if len(correct) >= 5 and len(incorrect) >= 5:
                    break

    # Afișăm exemple corecte
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Exemple de clasificare corectă și incorectă", fontsize=16)
    
    for i, (img, pred, label) in enumerate(correct):
        axs[0, i].imshow(img.squeeze(), cmap="gray")
        axs[0, i].set_title(f"Corect: {classes[label]} - Pred: {classes[pred]}", fontsize=10)
        axs[0, i].axis("off")

    # Afișăm exemple incorecte
    for i, (img, pred, label) in enumerate(incorrect):
        axs[1, i].imshow(img.squeeze(), cmap="gray")
        axs[1, i].set_title(f"Et: {classes[label]} - Pred: {classes[pred]}", fontsize=10)
        axs[1, i].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustăm layout-ul pentru a evita suprapunerea titlurilor
    plt.show()


# Executăm funcțiile de evaluare și afișare a rezultatelor
if __name__ == "__main__":
    evaluate(model, test_loader)
    display_classification_examples(model, test_loader, classes)
