import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


def display_images(
    images=None, grid=None, processed_image=None, titles=None, batch=None, predictions=None, labels=None, class_names=None, figsize=(10, 5)
):
    """
    Fonction polyvalente pour afficher des images dans divers formats.

    Cette fonction peut gérer plusieurs scénarios d'affichage :
    - Une comparaison côte à côte de deux images.
    - Une grille d'images depuis un tenseur.
    - Un seul tenseur d'image traité.
    - Un batch d'images avec leurs labels réels et prédits.

    Args:
        images (list, optional): Une liste contenant deux images PIL pour comparaison. Par défaut None.
        grid (torch.Tensor, optional): Un tenseur représentant une grille d'images. Par défaut None.
        processed_image (torch.Tensor, optional): Un seul tenseur d'image à afficher. Par défaut None.
        titles (tuple, optional): Un tuple de chaînes pour les titres des images comparées. Par défaut None.
        batch (torch.Tensor, optional): Un batch d'images représenté comme un tenseur de grille. Par défaut None.
        predictions (torch.Tensor, optional): Un tenseur de labels prédits pour un batch d'images. Par défaut None.
        labels (torch.Tensor, optional): Un tenseur de labels réels pour un batch d'images. Par défaut None.
        class_names (list, optional): Une liste de chaînes représentant les noms de classes. Par défaut None.
        figsize (tuple, optional): La taille de la figure matplotlib. Par défaut (10, 5).
    """

    def imshow(img_tensor):
        """Fonction utilitaire pour afficher un tenseur comme une image."""
        img = img_tensor / 2 + 0.5
        img = np.clip(img.numpy(), 0, 1)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.axis('off')

    if predictions is not None and labels is not None and class_names is not None:
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            num_images = images.size(0)
            fig, axes = plt.subplots(1, num_images, figsize=figsize)
            if num_images == 1:
                axes = [axes]
            for i, ax in enumerate(axes):
                plt.sca(ax)
                imshow(images[i])
                true_label = class_names[labels[i].item()]
                pred_label = class_names[predictions[i].item()]
                ax.set_title(f"Réel: {true_label}\nPréd: {pred_label}")
            plt.tight_layout()
            plt.show()

    elif batch is not None:
        if isinstance(batch, torch.Tensor):
            plt.figure(figsize=figsize)
            imshow(batch)
            plt.show()

    elif processed_image is not None:
        if isinstance(processed_image, torch.Tensor):
            plt.figure(figsize=figsize)
            plt.imshow(processed_image.permute(1, 2, 0))
            plt.axis('off')
            plt.show()

    elif grid is not None:
        if isinstance(grid, torch.Tensor):
            plt.figure(figsize=figsize)
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.show()

    elif images is not None and isinstance(images, list) and len(images) == 2 and all(isinstance(img, Image.Image) for img in images):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, img, title in zip(axes, images, (titles if titles else ["Image 1", "Image 2"])):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


def display_train_images(dataset_path):
    """
    Affiche une grille d'images sélectionnées aléatoirement depuis un dataset d'entraînement spécifié.

    Args:
        dataset_path: Le chemin racine vers le dataset.
    """
    paths = {
        "train/real": os.path.join(dataset_path, 'train', 'real'),
        "train/fake": os.path.join(dataset_path, 'train', 'fake')
    }

    images_to_plot = []
    valid_exts = ('.jpg', '.jpeg', '.png')

    for title, directory in paths.items():
        try:
            with os.scandir(directory) as entries:
                image_files = [
                    entry.path for entry in entries
                    if entry.is_file() and entry.name.lower().endswith(valid_exts)
                ]

            if len(image_files) < 3:
                print(f"Avertissement : Pas assez d'images dans '{directory}'. Trouvé {len(image_files)}, besoin de 3.")
                continue

            for img_path in random.sample(image_files, 3):
                images_to_plot.append((img_path, title))

        except FileNotFoundError:
            print(f"Erreur : Répertoire non trouvé à '{directory}'.")
            return

    if len(images_to_plot) != 6:
        print("Impossible de rassembler assez d'images à afficher. Abandon.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    thumbnail_size = (256, 256)

    for ax, (img_path, title) in zip(axes.ravel(), images_to_plot):
        try:
            with Image.open(img_path) as img:
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')

        except (UnidentifiedImageError, FileNotFoundError):
            ax.set_title(f"Erreur de chargement d'image\n{os.path.basename(img_path)}", color='red')
            ax.axis('off')

    plt.tight_layout()
    plt.show()
