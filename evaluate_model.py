import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_mobilenetv3_model(weights_path, num_classes=None):
    """
    Charge un modèle MobileNetV3-Large pré-entraîné depuis torchvision.

    Args:
        weights_path (str): Le chemin du fichier vers les poids du modèle .pth sauvegardés.
        num_classes (int, optional): Nombre de classes. Si fourni, adapte le classifier avant de charger.

    Returns:
        torch.nn.Module: Un modèle MobileNetV3-Large pré-entraîné.
    """
    # Charge le modèle MobileNetV3-Large pré-entraîné sans poids pré-entraînés.
    model = tv_models.mobilenet_v3_large(weights=None)

    # Si num_classes fourni, adapte le classifier
    if num_classes is not None:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    # Charge le dictionnaire d'état (poids) depuis le fichier local.
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    return model


# Charge le modèle entraîné en utilisant la fonction helper
trained_model_path = "./best_model_nanobanana.pth"
model = load_mobilenetv3_model(trained_model_path, num_classes=2)
model = model.to(DEVICE)
model.eval()

# Transformations
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Charge le dataset de validation
print("🔄 Chargement du dataset de validation...")
dataset_path = "./AIvsReal_nanobanana_pro"
val_dataset = ImageFolder(root=f"{dataset_path}/test", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"📊 Dataset: {len(val_dataset)} images")
print(f"   Classes: {val_dataset.classes}")
print(f"   Distribution: {[val_dataset.targets.count(i) for i in range(len(val_dataset.classes))]}\n")

# Évaluation
print("🔍 Évaluation en cours...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calcule les métriques
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("=" * 60)
print("📈 RÉSULTATS D'ÉVALUATION")
print("=" * 60)
print(f"\n✅ Accuracy:  {accuracy * 100:.2f}%")
print(f"✅ Precision: {precision * 100:.2f}%")
print(f"✅ Recall:    {recall * 100:.2f}%")
print(f"\n📊 Matrice de confusion:")
print(f"   Classes: {val_dataset.classes}")
print(f"   {cm}")
print("\n" + "=" * 60)

