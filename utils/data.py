import os

def display_data_loader_contents(data_loader):
    """Affiche le contenu du chargeur de données."""
    try:
        print("Nombre total d'images dans le dataset :", len(data_loader.dataset))
        print("Nombre total de lots :", len(data_loader))
        for batch_idx, (data, labels) in enumerate(data_loader):
            print(f"--- Lot {batch_idx + 1} ---")
            print(f"Forme des données : {data.shape}")
            print(f"Forme des labels : {labels.shape}")
            break
    except StopIteration:
        print("Le chargeur de données est vide.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")


def dataset_images_per_class(dataset_path):
    """
    Compte le nombre de fichiers image par classe dans une structure de répertoire de dataset.

    Args:
        dataset_path: Le chemin racine du dataset.
    """
    print(f"Analyse du dataset à : {dataset_path}\n")
    valid_exts = ('.jpg', '.jpeg', '.png')

    split_names = ['train', 'test']

    try:
        for split_name in split_names:
            split_path = os.path.join(dataset_path, split_name)

            if os.path.isdir(split_path):
                print(f"— {split_name.capitalize()} —")

                class_entries = sorted(os.scandir(split_path), key=lambda e: e.name)

                for class_entry in class_entries:
                    if class_entry.is_dir():
                        image_count = sum(1 for file_entry in os.scandir(class_entry.path)
                                         if file_entry.is_file() and file_entry.name.lower().endswith(valid_exts))

                        print(f"{class_entry.name.capitalize()}: {image_count} images")

                print()

    except FileNotFoundError:
        print(f"Erreur : Le répertoire '{dataset_path}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")
