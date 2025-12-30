import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import base64

app = FastAPI(title="FakeFinder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "./models/best_model_nanobanana_pro.pth"
REAL_THRESHOLD = 0.7
FAKE_THRESHOLD = 0.7

_model = None

def load_mobilenetv3_model(weights_path, num_classes=None):
    model = tv_models.mobilenet_v3_large(weights=None)
    if num_classes is not None:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def get_model():
    global _model
    if _model is None:
        _model = load_mobilenetv3_model(MODEL_PATH, num_classes=2)
        _model = _model.to(DEVICE)
        _model.eval()
    return _model

def predict_image(model, image_bytes):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = val_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    real_conf = float(probs[1].item())
    fake_conf = float(probs[0].item())

    if real_conf >= REAL_THRESHOLD:
        pred_label = "real"
        conf = real_conf
    elif fake_conf >= FAKE_THRESHOLD:
        pred_label = "fake"
        conf = fake_conf
    else:
        pred_idx = int(probs.argmax().item())
        pred_label = "real" if pred_idx == 1 else "fake"
        conf = float(probs[pred_idx].item())

    return {
        "label": pred_label,
        "confidence": round(conf * 100, 2),
        "real_confidence": round(real_conf * 100, 2),
        "fake_confidence": round(fake_conf * 100, 2)
    }

@app.get("/")
def home():
    return {"message": "FakeFinder API - Utilisez /predict pour analyser une image"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Fichier non supporté. Utilisez une image (jpg, png, webp)")

    try:
        image_bytes = await file.read()
        model = get_model()
        result = predict_image(model, image_bytes)

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        result["image_data"] = f"data:{file.content_type};base64,{image_base64}"

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse: {str(e)}")

