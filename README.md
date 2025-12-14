# Fake Image Finder - Nano Banana Pro

Détecteur d'images générées par IA utilisant un **transfer learning** avec MobileNetV3 fine-tuné pour identifier spécifiquement les images créées par **Nano Banana Pro** (modèle d'IA multimodale de Google).

## 🎯 Objectif

Ce projet vise à distinguer les images **réelles** des images **générées par IA**, avec un focus particulier sur la détection des images créées par Nano Banana Pro. Le modèle utilise une approche de **transfer learning** en deux étapes :

1. **Entraînement initial** : Détection générale d'images fake (Stable Diffusion, Midjourney, DALL-E)
2. **Fine-tuning** : Adaptation spécifique pour détecter les images Nano Banana Pro

## 🔄 Transfer Learning - Point Clé du Projet

Ce projet repose entièrement sur une stratégie de **transfer learning** en cascade :

### Étape 1 : Pré-entraînement ImageNet
- **Modèle de base** : MobileNetV3-Large pré-entraîné sur ImageNet
- **Connaissances transférées** : Features génériques de reconnaissance d'images (bords, textures, formes)

### Étape 2 : Transfer Learning vers la détection fake/real
- **Source** : Modèle ImageNet
- **Cible** : Détection générale d'images fake (SD, Midjourney, DALL-E)
- **Méthode** : Fine-tuning du classifier (features extractor gelé)
- **Résultat** : `best_model_midjourney_dalle_sd.pth`

### Étape 3 : Transfer Learning vers Nano Banana Pro
- **Source** : Modèle fine-tuné SD/Midjourney/DALL-E
- **Cible** : Détection spécifique Nano Banana Pro
- **Méthode** : Fine-tuning du classifier avec learning rate réduit (0.0005)
- **Résultat** : `best_model_nanobanana_pro.pth`

**Avantages du transfer learning** :
- ✅ Réutilisation des connaissances pré-existantes
- ✅ Entraînement rapide avec peu de données
- ✅ Meilleures performances que l'entraînement from scratch
- ✅ Adaptation progressive du modèle général vers le cas spécifique

## 🏗️ Architecture

- **Modèle de base** : MobileNetV3-Large (transfer learning depuis ImageNet)
- **Pré-entraînement** : ImageNet (1.4M images, 1000 classes)
- **Transfer learning** : Cascade en 3 étapes (ImageNet → Fake général → Nano Banana Pro)
- **Fine-tuning** : Classifier uniquement (features extractor gelé)
- **Classes** : 2 (Real / Fake)
- **Résolution d'entrée** : 224x224

## 🚀 Utilisation

### Installation

```bash
# Installation des dépendances avec uv
uv sync
```

### Téléchargement des Datasets

Après l'installation, téléchargez les deux datasets depuis Hugging Face :

```bash
# Dataset Midjourney, DALL-E, Stable Diffusion
uv run python download_dataset_images.py julienlucas/midjourney-dalle-sd-dataset ./AIvsReal_midjourney_dalle_sd

# Dataset Nano Banana Pro
uv run python download_dataset_images.py julienlucas/nanobanana-pro-dataset ./AIvsReal_nanobanana_pro
```

Le script `download_dataset_images.py` télécharge automatiquement les fichiers Parquet depuis Hugging Face, extrait les images dans la structure `train/real`, `train/fake`, `test/real`, `test/fake`, puis nettoie les fichiers temporaires.

### Entraînement

#### 1. Transfer Learning initial (SD, Midjourney, DALL-E)

```bash
uv run python finetune_midjourney_dalle_sd.py
```

**Transfer learning** depuis ImageNet vers la détection générale d'images fake.
Génère `models/best_model_midjourney_dalle_sd.pth` - modèle de base pour détecter les images fake générales.

#### 2. Transfer Learning vers Nano Banana Pro

```bash
uv run python finetune_nanobananapro.py
```

**Transfer learning** depuis le modèle SD/Midjourney/DALL-E vers Nano Banana Pro.
Génère `models/best_model_nanobanana_pro.pth` - modèle adapté pour Nano Banana Pro.

**Configuration du fine-tuning :**
- Learning rate : 0.0005
- Batch size : 32
- Epochs : 1 (convergence rapide)
- Data augmentation : RandomResizedCrop, flips, rotations, color jitter, perspective

### Inférence

#### Inférence simple avec visualisation Grad-CAM

```bash
uv run python inference.py
```

Affiche la prédiction et les régions importantes de l'image.

#### Évaluation complète du dataset de test

```bash
uv run python inference_check_test_dataset.py
```

Teste toutes les images du dataset `test/real` et `test/fake` et affiche :
- Précision, Recall, F1-Score par classe
- Accuracy globale
- Statistiques détaillées

## 📊 Performances

### Modèle fine-tuné Nano Banana Pro

- **Accuracy globale** : ~89-90%
- **Précision REAL** : ~89%
- **Recall REAL** : ~89%
- **Précision FAKE** : ~89%
- **Recall FAKE** : ~89%

### Dataset

- **Train** : 2250 images fake Nano Banana Pro + images real
- **Test** : 500 images fake Nano Banana Pro + images real
- **Ratio** : ~82% train / 18% test

## 📥 Sources des Images Nano Banana Pro

Les images Nano Banana Pro utilisées pour l'entraînement ont été collectées depuis :

- **[YouMind](https://youmind.com/fr-FR/nano-banana-pro-prompts)** - Collection de prompts et images Nano Banana Pro
- **[Higgsfield.ai](https://higgsfield.ai/nano-banana-pro-preview)** - Aperçu et exemples Nano Banana Pro
- **[Awesome Nano Banana Pro (GitHub)](https://github.com/ZeroLu/awesome-nanobanana-pro)** - Collection open-source d'exemples
- **[PromptGather.io](https://promptgather.io)** - Plateforme de collecte de prompts Nano Banana Pro
- **[Google Sheets - PromptGather](https://docs.google.com/spreadsheets/d/1GAp_yaqAX9y_K8lnGQw9pe_BTpHZehoonaxi4whEQIE/edit?gid=116507383#gid=116507383)** - Base de données de prompts avec images

## 🔧 Configuration

### Transformations d'entraînement

- `RandomResizedCrop(224, 224)` - scale (0.7, 1.0)
- `RandomHorizontalFlip` - p=0.5
- `RandomVerticalFlip` - p=0.2
- `RandomRotation` - degrees=20
- `ColorJitter` - brightness, contrast, saturation, hue
- `RandomAffine` - translate, scale
- `RandomPerspective` - p=0.3

### Transformations de validation

- `Resize(256, 256)`
- `CenterCrop(224)`
- Normalisation ImageNet

## 📝 Notes Techniques

- **Approche** : Transfer Learning en cascade (ImageNet → Fake général → Nano Banana Pro)
- **Device** : MPS (Apple Silicon) ou CPU
- **Framework** : PyTorch
- **Optimiseur** : Adam (lr=0.0005)
- **Loss** : CrossEntropyLoss
- **Seuils de confiance** : 0.7 pour REAL et FAKE

## 🎨 Fonctionnalités

- ✅ **Transfer Learning** en cascade (ImageNet → Fake général → Nano Banana Pro)
- ✅ Détection d'images fake/real
- ✅ Visualisation Grad-CAM pour comprendre les décisions
- ✅ Fine-tuning spécifique Nano Banana Pro
- ✅ Évaluation complète avec métriques détaillées
- ✅ Support des formats : JPG, PNG, WebP

## 📄 Licence

Ce projet est destiné à la recherche et à l'éducation sur la détection d'images générées par IA.
