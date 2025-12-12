import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

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

# Seuil de confiance pour déclarer une image comme "real" ou "fake" (0.7 = 70%)
REAL_THRESHOLD = 0.7
FAKE_THRESHOLD = 0.7

# Charge toutes les images
real_dir = "./AIvsReal_nanobanana_pro/test/real"
all_images = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

real_detected = []
real_confidences = []

for img_name in all_images:
    img_path = os.path.join(real_dir, img_name)

    pil_image = Image.open(img_path).convert("RGB")
    input_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        real_conf = float(probs[1].item())
        fake_conf = float(probs[0].item())

    # Utilise les seuils au lieu de argmax simple
    if real_conf >= REAL_THRESHOLD:
        is_real = True
    elif fake_conf >= FAKE_THRESHOLD:
        is_real = False
    else:
        # Si aucun seuil n'est atteint, utilise la classe avec la proba la plus élevée
        is_real = real_conf > fake_conf

    status = "✅ REAL" if is_real else "❌ FAKE"
    print(f"{img_name:30s} | {status:10s} | Real: {real_conf*100:5.1f}% | Fake: {fake_conf*100:5.1f}%")

    if is_real:
        real_detected.append(img_name)
        real_confidences.append(real_conf)

print("=" * 60)
print(f"\n📊 Résultats (seuil {REAL_THRESHOLD*100:.0f}%):")
print(f"   Images détectées comme REAL: {len(real_detected)}/{len(all_images)} ({len(real_detected)/len(all_images)*100:.1f}%)")
if real_confidences:
    avg_conf = sum(real_confidences) / len(real_confidences)
    print(f"   Confiance moyenne (REAL détectées): {avg_conf*100:.1f}%")
else:
    print(f"   Confiance moyenne (REAL détectées): 0.0%")

