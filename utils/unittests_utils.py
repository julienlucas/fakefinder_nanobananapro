import inspect
import math
import os
import re
import optuna
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
from typing import Union, List, Type
from collections import OrderedDict, defaultdict

import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, Subset, dataloader, sampler



def remove_comments(code):
    # Ce motif regex correspond aux commentaires dans le code
    pattern = r'#.*'

    # Utiliser re.sub() pour remplacer les commentaires par une chaîne vide
    code_without_comments = re.sub(pattern, '', code)

    # Diviser le code en lignes, supprimer les espaces de chaque ligne et filtrer les lignes vides
    lines = code_without_comments.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]

    # Joindre les lignes non vides en une seule chaîne
    return '\n'.join(non_empty_lines)


def load_rows(path_to_csv, row_range=(20, 29)):
    """
    Charge un fichier CSV et retourne une plage spécifique de lignes.

    Args:
        path_to_csv (str): Le chemin du fichier CSV.
        row_range (tuple): Un tuple de deux entiers (début, fin)
                           spécifiant la plage de lignes inclusive à charger.

    Returns:
        pd.DataFrame: Un DataFrame pandas contenant les lignes spécifiées.
    """
    # Déballer le début et la fin du tuple de plage
    start_row, end_row = row_range

    # Charger le fichier CSV entier dans un DataFrame
    df = pd.read_csv(path_to_csv)

    # Sélectionner les lignes en utilisant .iloc pour l'indexation basée sur la position entière.
    # Ajouter 1 à end_row pour rendre la tranche inclusive.
    selected_df = df.iloc[start_row:end_row + 1]

    return selected_df


def get_train_test():
    # Définir le chemin où les données EMNIST seront stockées
    data_path = "./EMNIST_data"

    # Vérifier si le dossier de données existe pour éviter le re-téléchargement
    if os.path.exists(data_path) and os.path.isdir(data_path):
        download = False
    else:
        download = True

    # Moyenne et écart-type précalculés pour le dataset EMNIST Letters
    mean = (0.1736,)
    std = (0.3317,)

    # Créer une transformation qui convertit les images en tenseurs et les normalise
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertit les images en tenseurs PyTorch et met à l'échelle les valeurs de pixels à [0, 1]
        transforms.Normalize(
            mean=mean, std=std
        ),  # Applique la normalisation en utilisant la moyenne et l'écart-type calculés
    ])

    # Charger le dataset d'entraînement EMNIST Letters
    train_dataset = datasets.EMNIST(
        root=data_path,
        split="letters",
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.EMNIST(
        root=data_path,
        split="letters",
        train=False,
        download=False,
        transform=transform,
    )
    return train_dataset, test_dataset


# region = Structure générale de vérification =


class TestBattery:
    def __init__(self, learner_object):
        self.learner_object = learner_object

        self._get_reference_inputs()
        self.extract_info()
        self.get_reference_checks()

    def _get_reference_inputs(self):
        pass

    def extract_info(self):
        pass

    def get_reference_checks(self):
        pass

    def _create_reference_checks(self):
        checks_dict = {}
        if self.reference_checks is not None:
            for key in self.reference_checks.keys():
                check_fcn = getattr(self, f"{key}", None)
                # exécuter la méthode correspondante
                got, want, failed = check_fcn()
                checks_dict[key] = got

        return checks_dict

    def _check(self, check_name, got):
        want = self.reference_checks[check_name]
        condition = got != want
        return got, want, condition


def check_method_call_init(class_name, method_name, *args, **kwargs):
    with patch.object(class_name, method_name, autospec=True) as mocked:
        obj = class_name(*args, **kwargs)
        return mocked.called


def check_method_call(
    class_name,
    outer_method,
    inner_method,
    init_args=None,
    init_kwargs=None,
    outer_args=None,
    outer_kwargs=None,
):
    """
    Vérifie si l'appel de outer_method() sur une instance de class_name
    entraîne l'appel de inner_method.

    - init_args/init_kwargs: arguments pour l'initialisation de la classe
    - outer_args/outer_kwargs: arguments pour outer_method
    """
    init_args = init_args or ()
    init_kwargs = init_kwargs or {}
    outer_args = outer_args or ()
    outer_kwargs = outer_kwargs or {}

    with patch.object(class_name, inner_method, autospec=True) as mocked:
        obj = class_name(*init_args, **init_kwargs)
        getattr(obj, outer_method)(*outer_args, **outer_kwargs)
        return mocked.called


# region = Exercice 1 =
def check_shuffle(data_loader, should_shuffle):
    """
    Vérifie si un DataLoader mélange les données en observant l'ordre des labels.
    """
    sampler = data_loader.sampler
    if should_shuffle:
        return isinstance(sampler, RandomSampler)
    else:
        return isinstance(sampler, SequentialSampler)


class DataLoaderBattery(TestBattery):

    def _get_reference_inputs(self):
        self.train_dataset, self.test_dataset = get_train_test()
        self.batch_size = 32

    def extract_info(self):
        self.train_loader, self.test_loader = self.learner_object(
            self.train_dataset, self.test_dataset, batch_size=self.batch_size
        )

    def get_reference_checks(self):
        self.reference_checks = {
            # "train_loader_type": None,
            # "test_loader_type": None,
            "train_loader_batch_size": 32,
            "test_loader_batch_size": 32,
            "train_loader_length": 3900,
            "test_loader_length": 650,
            "train_loader_shuffle": True,
            "test_loader_shuffle": False,
        }

    def train_loader_batch_size(self):

        got = self.train_loader.batch_size

        name_check = "train_loader_batch_size"
        return self._check(name_check, got)

    def test_loader_batch_size(self):
        got = self.test_loader.batch_size

        name_check = "test_loader_batch_size"
        return self._check(name_check, got)

    def train_loader_length(self):
        got = len(self.train_loader)

        name_check = "train_loader_length"
        return self._check(name_check, got)

    def test_loader_length(self):
        got = len(self.test_loader)

        name_check = "test_loader_length"
        return self._check(name_check, got)

    def train_loader_shuffle(self):
        got = check_shuffle(self.train_loader, should_shuffle=True)

        name_check = "train_loader_shuffle"
        return self._check(name_check, got)

    def test_loader_shuffle(self):
        got = check_shuffle(self.test_loader, should_shuffle=True)

        name_check = "test_loader_shuffle"
        return self._check(name_check, got)


# endregion =


# region = Exercice 2 =
class ModelBattery(TestBattery):
    def _get_reference_inputs(self):
        self.num_classes = 26
        self.input_size = 784  # 28*28

    def extract_info(self):
        self.model, self.loss_function, self.optimizer = self.learner_object(
            num_classes=self.num_classes
        )

    def get_reference_checks(self):
        self.reference_checks = {
            "model_num_layers": True,
            "model_type": nn.Sequential,
            "first_layer_type": nn.Flatten,
            "last_layer_type": nn.Linear,
            "loss_function_type": nn.CrossEntropyLoss,
            "optimizer_type": optim.Adam,
            "learning_rate": 0.001,
            "num_classes": self.num_classes,
        }

    def model_type(self):
        got = type(self.model)

        name_check = "model_type"
        return self._check(name_check, got)

    def model_num_layers(self):
        got = 1 <= len(self.model) <= 7

        name_check = "model_num_layers"
        return self._check(name_check, got)

    def first_layer_type(self):
        first_layer = self.model[0]
        got = type(first_layer)

        name_check = "first_layer_type"
        return self._check(name_check, got)

    def last_layer_type(self):
        last_layer = self.model[-1]
        got = type(last_layer)

        name_check = "last_layer_type"
        return self._check(name_check, got)

    def middle_layers_type(self):
        middle_layers = self.model[1:-1]
        checks = []

        for layer in middle_layers:
            got = type(layer)
            want = [nn.Linear, nn.ReLU]
            failed = got not in want
            checks.append((got, want, failed))

        return checks

    def loss_function_type(self):
        got = type(self.loss_function)

        name_check = "loss_function_type"
        return self._check(name_check, got)

    def optimizer_type(self):
        got = type(self.optimizer)

        name_check = "optimizer_type"
        return self._check(name_check, got)

    def learning_rate(self):
        got = self.optimizer.defaults["lr"]

        name_check = "learning_rate"
        return self._check(name_check, got)

    def hidden_layers_inputs(self):
        checks = []
        input_size = self.input_size

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                got = layer.in_features
                want = input_size
                failed = got != want
                checks.append((got, want, failed))
            input_size = layer.out_features
        return checks

    def hidden_layers_outputs(self):
        last_layer = self.model[-1]
        checks = []
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                got = layer.out_features
                want = True
                failed = (got <= 0) or (layer == last_layer)
                checks.append((got, want, failed))
        return checks

    def num_classes(self):
        last_layer = self.model[-1]
        got = last_layer.out_features

        name_check = "num_classes"
        return self._check(name_check, got)


# endregion =


# region = Exercice 3 =
def remove_comments_bis(code):
    # Ce motif regex correspond aux commentaires dans le code
    pattern = r"#.*"

    # Utiliser re.sub() pour remplacer les commentaires par une chaîne vide
    code_without_comments = re.sub(pattern, "", code)

    # Diviser le code en lignes, supprimer les espaces de chaque ligne et filtrer les lignes vides
    lines = code_without_comments.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]

    # Joindre les lignes non vides en une seule chaîne
    return "\n".join(non_empty_lines)


class TrainBattery(TestBattery):
    def __init__(self, learner_object, model, loss_function, optimizer, train_loader):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader

        super().__init__(learner_object)

    def _get_reference_inputs(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inputs = torch.randn(2, 1, 28, 28).to(self.device)

        self.inputs_1_sample = torch.randn(1, 1, 28, 28).to(self.device)

    def extract_info(self):
        self.trained_model, self.loss_0 = self.learner_object(
            self.model,
            self.loss_function,
            self.optimizer,
            self.train_loader,
            device=self.device,
            verbose=False,
        )

        with torch.no_grad():
            self.outputs = self.model(self.inputs)
            self.outputs_1_sample = self.model(self.inputs_1_sample)

    def get_reference_checks(self):
        self.reference_checks = {
            "model_type": nn.Sequential,
            "batch_size": 2,
        }

    # pas de référence pour celui-ci
    def required_methods_check(self):
        checks = []

        source_code = inspect.getsource(self.learner_object)
        source_code = remove_comments(source_code)

        required_methods = [
            "optimizer.zero_grad()",
            "loss.backward()",
            "optimizer.step()",
        ]

        for method in required_methods:
            got = method in source_code
            want = method
            failed = not got
            checks.append((got, want, failed))
        return checks

    def model_type(self):
        got = type(self.trained_model)
        name_check = "model_type"
        return self._check(name_check, got)

    # pas de référence pour celui-ci
    def train_check(self):
        # Entraîner une deuxième fois et vérifier que la perte diminue
        second_train, loss_1 = self.learner_object(
            self.trained_model,
            self.loss_function,
            self.optimizer,
            self.train_loader,
            device=self.device,
            verbose=False,
        )

        got = (self.loss_0, loss_1)

        rel_tol = 1e-5

        want = None

        failed = math.isclose(loss_1, self.loss_0, rel_tol=rel_tol)

        return got, want, failed

    def batch_size(self):
        got = self.outputs.shape[0]
        name_check = "batch_size"
        return self._check(name_check, got)

    def output_shape(self):
        got = self.outputs.shape
        want = torch.Size([2, 26])
        failed = len(got) != len(want) or (got[1] != want[1])
        return got, want, failed

    def batch_size_1_sample(self):
        got = self.outputs_1_sample.shape[0]
        want = 1
        failed = got != want
        return got, want, failed

    def output_shape_1_sample(self):
        got = self.outputs_1_sample.shape
        want = torch.Size([1, 26])
        failed = len(got) != len(want) or (got[1] != want[1])
        return got, want, failed


# endregion =

# region = Exercice 4 =
class EvaluateBattery(TestBattery):
    def __init__(self, learner_object, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        super().__init__(learner_object)

    def _get_reference_inputs(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inputs = torch.randn(2, 1, 28, 28).to(self.device)

        self.inputs_1_sample = torch.randn(1, 1, 28, 28).to(self.device)

    def extract_info(self):
        self.model = self.model.to(self.device)

        # Appliquer la fonction d'évaluation pour obtenir la précision
        self.accuracy = self.learner_object(
            self.model, self.data_loader, device=self.device, verbose=False
        )

        with torch.no_grad():
            self.outputs = self.model(self.inputs)
            self.outputs_1_sample = self.model(self.inputs_1_sample)

    def get_reference_checks(self):
        self.reference_checks = {
            "no_grad": None,
            "output_shape": torch.Size([2, 26]),
            "output_shape_1_sample": torch.Size([1, 26]),
        }

    def no_grad_present(self):
        source_code = inspect.getsource(self.learner_object)
        source_code = remove_comments(source_code)

        got = "with torch.no_grad()" in source_code
        want = None
        failed = not got
        return got, want, failed

    def output_shape(self):
        got = self.outputs.shape
        name_check = "output_shape"
        return self._check(name_check, got)

    def output_shape_1_sample(self):
        got = self.outputs_1_sample.shape
        name_check = "output_shape_1_sample"
        return self._check(name_check, got)




# region = Exercice 1 - CustomDataset =
class DatasetBattery(TestBattery):
    def __init__(self, learner_class):
        super().__init__(learner_class)

    def _get_reference_inputs(self):
        self.root_dir = "./plants_dataset"

        # à utiliser dans certains tests
        self.main_transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )

    def extract_info(self):
        # créer le dataset
        self.learner_dataset = self.learner_object(
            root_dir=self.root_dir, transform=None
        )

        self.ref_idx = 1000

    def get_reference_checks(self):
        self.reference_checks = {
            "labels_len": 3000,
            "label_particular": 10,
            "class_names_len": 30,
            "description_particular": "galangal",
            "len_method": 3000,
            "get_item_img_shape": (673, 379),
            "get_item_img_type": PIL.Image.Image,
            "get_item_label": 10,
            "get_item_transform": (3, 128, 128),
            "call_load_labels": True,
            "call_read_classname": True,
            "call_retrieve_image": True,
        }

    def call_load_labels(self):
        got = check_method_call_init(
            class_name=self.learner_object,
            method_name="load_labels",
            root_dir=self.root_dir,
        )
        name_check = "call_load_labels"
        return self._check(name_check, got)

    def call_read_classname(self):
        got = check_method_call_init(
            class_name=self.learner_object,
            method_name="read_classname",
            root_dir=self.root_dir,
        )
        name_check = "call_read_classname"
        return self._check(name_check, got)

    def labels_len(self):
        learner_labels = self.learner_dataset.labels
        got = len(learner_labels)

        name_check = "labels_len"
        return self._check(name_check, got)

    def label_particular(self):
        got = self.learner_dataset.labels[self.ref_idx]

        name_check = "label_particular"
        return self._check(name_check, got)

    def class_names_len(self):
        got = len(self.learner_dataset.class_names)

        name_check = "class_names_len"
        return self._check(name_check, got)

    def description_particular(self):
        label = self.learner_dataset.labels[self.ref_idx]
        got = self.learner_dataset.get_label_description(label)
        name_check = "description_particular"
        return self._check(name_check, got)

    def len_method(self):
        got = len(self.learner_dataset)
        name_check = "len_method"
        return self._check(name_check, got)

    def call_retrieve_image(self):
        got = check_method_call(
            class_name=self.learner_object,
            outer_method="__getitem__",
            inner_method="retrieve_image",
            init_args=(self.root_dir,),
            outer_args=(self.ref_idx,),
        )
        name_check = "call_retrieve_image"
        return self._check(name_check, got)

    def get_item_img_shape(self):
        img, label = self.learner_dataset[self.ref_idx]
        got = img.size  # La taille de l'image PIL est (largeur, hauteur)
        name_check = "get_item_img_shape"
        return self._check(name_check, got)

    def get_item_img_type(self):
        img, label = self.learner_dataset[self.ref_idx]
        got = type(img)  # La taille de l'image PIL est (largeur, hauteur)
        name_check = "get_item_img_type"
        return self._check(name_check, got)

    def get_item_label(self):
        img, label = self.learner_dataset[self.ref_idx]
        got = label
        name_check = "get_item_label"
        return self._check(name_check, got)

    def get_item_transform(self):
        self.learner_dataset.transform = self.main_transform
        img, label = self.learner_dataset[self.ref_idx]
        got = img.shape  # La forme du tenseur torch est (canaux, hauteur, largeur)
        name_check = "get_item_transform"
        return self._check(name_check, got)




# region = Exercice 2 - get_transformations =
class TransformationsBattery(TestBattery):
    def __init__(self, learner_function):
        super().__init__(learner_function)

    def _get_reference_inputs(self):
        self.mean = [0.6659, 0.6203, 0.4784]
        self.std = [0.2119, 0.2155, 0.2567]

    def extract_info(self):
        self.main_transform_learner, self.augm_transform_learner = self.learner_object(
            self.mean, self.std
        )

    def get_reference_checks(self):
        self.reference_checks = {
            "main_transform_len": 3,
            "first_transform_main": (transforms.Resize, (128, 128)),
            "second_transform_main": transforms.ToTensor,
            "third_transform_main": (
                transforms.Normalize,
                self.mean,
                self.std,
            ),
            "augm_transform_len": 5,
            "first_transform_augm": (transforms.RandomVerticalFlip, 0.5),
            "second_transform_augm": (transforms.RandomRotation, [-15, 15]),
            "third_transform_augm": (transforms.Resize, (128, 128)),
            "fourth_transform_augm": transforms.ToTensor,
            "fifth_transform_augm": (
                transforms.Normalize,
                self.mean,
                self.std,
            ),
        }

    def main_transform_len(self):
        got = len(self.main_transform_learner.transforms)
        name_check = "main_transform_len"
        return self._check(name_check, got)

    def first_transform_main(self):
        first_transform = self.main_transform_learner.transforms[0]
        type_transform = type(first_transform)
        size = first_transform.size
        got = (type_transform, size)
        name_check = "first_transform_main"
        return self._check(name_check, got)

    def second_transform_main(self):
        second_transform = self.main_transform_learner.transforms[1]
        type_transform = type(second_transform)
        got = type_transform
        name_check = "second_transform_main"
        return self._check(name_check, got)

    def third_transform_main(self):
        third_transform = self.main_transform_learner.transforms[2]
        type_transform = type(third_transform)
        mean_transform = third_transform.mean
        std_transform = third_transform.std
        got = (type_transform, mean_transform, std_transform)
        name_check = "third_transform_main"
        return self._check(name_check, got)

    def augm_transform_len(self):
        got = len(self.augm_transform_learner.transforms)
        name_check = "augm_transform_len"
        return self._check(name_check, got)

    def first_transform_augm(self):
        first_transform = self.augm_transform_learner.transforms[0]
        type_transform = type(first_transform)
        p = first_transform.p
        got = (type_transform, p)
        name_check = "first_transform_augm"
        return self._check(name_check, got)

    def second_transform_augm(self):
        second_transform = self.augm_transform_learner.transforms[1]
        type_transform = type(second_transform)
        degrees = second_transform.degrees
        got = (type_transform, degrees)
        name_check = "second_transform_augm"
        return self._check(name_check, got)

    def third_transform_augm(self):
        third_transform = self.augm_transform_learner.transforms[2]
        type_transform = type(third_transform)
        size = third_transform.size
        got = (type_transform, size)
        name_check = "third_transform_augm"
        return self._check(name_check, got)

    def fourth_transform_augm(self):
        fourth_transform = self.augm_transform_learner.transforms[3]
        type_transform = type(fourth_transform)
        got = type_transform
        name_check = "fourth_transform_augm"
        return self._check(name_check, got)

    def fifth_transform_augm(self):
        fifth_transform = self.augm_transform_learner.transforms[4]
        type_transform = type(fifth_transform)
        mean_transform = fifth_transform.mean
        std_transform = fifth_transform.std
        got = (type_transform, mean_transform, std_transform)
        name_check = "fifth_transform_augm"
        return self._check(name_check, got)



class DataloadersBattery(TestBattery):
    def __init__(self, learner_function, learner_dataset):
        self.learner_dataset = learner_dataset
        super().__init__(learner_function)

    def _get_reference_inputs(self):
        self.main_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.6659, 0.6203, 0.4784],
                    std=[0.2119, 0.2155, 0.2567],
                ),
            ]
        )

        self.augm_transform = transforms.Compose(
            [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.6659, 0.6203, 0.4784],
                    std=[0.2119, 0.2155, 0.2567],
                ),
            ]
        )

        self.batch_size = 64
        self.val_fraction = 0.2
        self.test_fraction = 0.1

    def extract_info(self):
        self.lrnr_train_loader, self.lrnr_val_loader, self.lrnr_test_loader = (
            self.learner_object(
                dataset=self.learner_dataset,
                batch_size=self.batch_size,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                main_transform=self.main_transform,
                augmentation_transform=self.augm_transform,
            )
        )

    def get_reference_checks(self):
        self.reference_checks = {
            "loaders_type": (dataloader.DataLoader,) * 3,
            "split_sizes": (2100, 600, 300),
            "random_split": (
                Subset,
                Subset,
                Subset,
            ),
            "train_transform": self.augm_transform,
            "test_transform": self.main_transform,
            "train_shuffle": sampler.RandomSampler,
            "val_shuffle": sampler.SequentialSampler,
            "test_shuffle": sampler.SequentialSampler,
        }

    def loaders_type(self):
        got = (
            type(self.lrnr_train_loader),
            type(self.lrnr_val_loader),
            type(self.lrnr_test_loader),
        )
        name_check = "loaders_type"
        return self._check(name_check, got)

    def split_sizes(self):
        train_size = len(self.lrnr_train_loader.dataset)
        val_size = len(self.lrnr_val_loader.dataset)
        test_size = len(self.lrnr_test_loader.dataset)
        got = (train_size, val_size, test_size)
        name_check = "split_sizes"
        return self._check(name_check, got)

    def random_split_and_subset(self):
        train_dataset_split = self.lrnr_train_loader.dataset.subset
        val_dataset_split = self.lrnr_val_loader.dataset.subset
        test_dataset_split = self.lrnr_test_loader.dataset.subset

        got = (
            type(train_dataset_split),
            type(val_dataset_split),
            type(test_dataset_split),
        )
        name_check = "random_split"
        return self._check(name_check, got)

    def train_transform(self):
        train_dataset_transform = self.lrnr_train_loader.dataset.transform
        got = train_dataset_transform
        name_check = "train_transform"
        return self._check(name_check, got)

    def test_transform(self):
        test_dataset_transform = self.lrnr_test_loader.dataset.transform
        got = test_dataset_transform
        name_check = "test_transform"
        return self._check(name_check, got)

    def train_shuffle(self):
        got = type(self.lrnr_train_loader.sampler)
        name_check = "train_shuffle"
        return self._check(name_check, got)

    def val_shuffle(self):
        got = type(self.lrnr_val_loader.sampler)
        name_check = "val_shuffle"
        return self._check(name_check, got)

    def test_shuffle(self):
        got = type(self.lrnr_test_loader.sampler)
        name_check = "test_shuffle"
        return self._check(name_check, got)


# region === Exercice 1 ===
def count_specific_layers(
    model: nn.Module, layer_types: Union[Type[nn.Module], List[Type[nn.Module]]]
) -> int:
    """Compte les types spécifiques de couches dans un modèle."""
    # Initialiser le compteur de couches
    layer_count = 0

    # Itérer sur tous les modules du modèle
    for module in model.modules():
        # Vérifier si le module est une instance du(des) type(s) de couche spécifié(s)
        if isinstance(module, layer_types):
            # Incrémenter le compteur si correspondance
            layer_count += 1

    # Retourner le compte total
    # print(f"Total layers of type {layer_types}: {layer_count}")
    return layer_count


def get_mock_params(case: str) -> dict:
    if case == "1":
        mock_params_model = {
            "n_layers": 2,
            "n_filters": [16, 32],
            "kernel_sizes": [3, 3],
            "dropout_rate": 0.1,
            "fc_size": 64,
            "num_classes": 2,
        }

        mock_params_data_loader = {
            "resolution": 32,
            "batch_size": 8,
        }

    # expected_features_layers = {'Conv2d_0': torch.Size([1, 16, 32, 32]), 'BatchNorm2d_0': torch.Size([1, 16, 32, 32]), 'ReLU_0': torch.Size([1, 16, 32, 32]), 'MaxPool2d_0': torch.Size([1, 16, 16, 16]), 'Conv2d_1': torch.Size([1, 32, 16, 16]), 'BatchNorm2d_1': torch.Size([1, 32, 16, 16]), 'ReLU_1': torch.Size([1, 32, 16, 16]), 'MaxPool2d_1': torch.Size([1, 32, 8, 8])}
    expected_features_params = {
        "Conv2d": {
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "bias": True,
        },
        "BatchNorm2d": {
            "num_features": 16,
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True,
        },
        "ReLU": {"inplace": False},
        "MaxPool2d": {
            "kernel_size": 2,
            "stride": 2,
            "padding": 0,
            "dilation": 1,
            "return_indices": False,
            "ceil_mode": False,
        },
    }

    # expected_classifier_shapes =  OrderedDict([('L0_Dropout', ((1, 2048), 'Dropout')), ('L1_Linear', ((1, 64), 'Linear')), ('L2_ReLU', ((1, 64), 'ReLU')), ('L3_Dropout', ((1, 64), 'Dropout')), ('L4_Linear', ((1, 2), 'Linear'))])
    expected_classifier_params = [
        ("Dropout", {"p": 0.1, "inplace": False}),
        ("Linear", {"in_features": 2048, "out_features": 64, "bias": True}),
        ("ReLU", {"inplace": False}),
        ("Linear", {"in_features": 64, "out_features": 2, "bias": True}),
    ]

    # {'Conv2d_0': torch.Size([1, 16, 32, 32]), 'BatchNorm2d_0': torch.Size([1, 16, 32, 32]), 'ReLU_0': torch.Size([1, 16, 32, 32]), 'MaxPool2d_0': torch.Size([1, 16, 16, 16]), 'Conv2d_1': torch.Size([1, 32, 16, 16]), 'BatchNorm2d_1': torch.Size([1, 32, 16, 16]), 'ReLU_1': torch.Size([1, 32, 16, 16]), 'MaxPool2d_1': torch.Size([1, 32, 8, 8])}
    # OrderedDict([('dropout_c_1', ((1, 2048), 'Dropout')), ('f_c_1', ((1, 64), 'Linear')), ('relu_c', ((1, 64), 'ReLU')), ('dropout_c_2', ((1, 64), 'Dropout')), ('f_c_2', ((1, 2), 'Linear'))])

    total_layers = (
        mock_params_model["n_layers"] * 4
    )  # Conv2d + BatchNorm2d + ReLU + MaxPool2d
    x_input = torch.randn(
        1,
        3,
        mock_params_data_loader["resolution"],
        mock_params_data_loader["resolution"],
    )

    # return mock_params_model, total_layers, expected_features_layers, x_input, expected_classifier_shapes
    return (
        mock_params_model,
        total_layers,
        expected_features_params,
        x_input,
        expected_classifier_params,
    )


def get_layer_output(name, layer_outputs):
    def hook(module, input, output):
        layer_outputs[name] = output.shape

    return hook


def add_hooks_features(model_features: nn.Module, layer_outputs: dict = None):
    for idx, block in enumerate(model_features.features):
        for layer in block:
            name = layer._get_name()
            layer.register_forward_hook(
                get_layer_output(name + f"_{idx}", layer_outputs)
            )


def get_sequence_features_shapes(model: nn.Module, x_input: torch.Tensor) -> dict:
    """
    Obtient les formes des features de chaque couche du modèle.
    """
    layer_outputs = {}  # Réinitialiser le dictionnaire global pour stocker les sorties
    add_hooks_features(model, layer_outputs=layer_outputs)

    # Effectuer une passe avant pour déclencher les hooks
    with torch.no_grad():
        _ = model(x_input)  # Exemple de forme d'entrée

    return layer_outputs


def get_layer_shapes_and_types_sequential(
    sequential: nn.Sequential, input_tensor: torch.Tensor
):
    """
    Retourne un OrderedDict mappant les noms de couches à (forme_sortie, type_couche).
    Si une couche a un nom comme '0', '1', etc. (par défaut de Sequential), il sera remplacé par
    '<TypeCouche>_<index>'. Sinon, le nom fourni est utilisé.
    """
    output_shapes = OrderedDict()

    def register_hook(name, layer):
        def hook(module, input, output):
            output_shapes[name] = (tuple(output.shape), type(module).__name__)

        layer.register_forward_hook(hook)

    for idx, (name, layer) in enumerate(sequential._modules.items()):
        if name.isdigit():
            layer_name = f"L{idx}_{type(layer).__name__}"
        else:
            layer_name = name
        register_hook(layer_name, layer)

    with torch.no_grad():
        sequential(input_tensor)

    return output_shapes

    # region == Extraction des hyperparamètres pour chaque couche ==


def get_layer_hyperparams(layer):
    if isinstance(layer, torch.nn.Conv2d):
        return {
            "in_channels": layer.in_channels,
            "out_channels": layer.out_channels,
            "kernel_size": layer.kernel_size,
            "stride": layer.stride,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "groups": layer.groups,
            "bias": layer.bias is not None,
        }
    elif isinstance(layer, torch.nn.Linear):
        return {
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "bias": layer.bias is not None,
        }
    elif isinstance(layer, torch.nn.BatchNorm2d):
        return {
            "num_features": layer.num_features,
            "eps": layer.eps,
            "momentum": layer.momentum,
            "affine": layer.affine,
            "track_running_stats": layer.track_running_stats,
        }
    elif isinstance(layer, torch.nn.MaxPool2d):
        return {
            "kernel_size": layer.kernel_size,
            "stride": layer.stride,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "return_indices": layer.return_indices,
            "ceil_mode": layer.ceil_mode,
        }
    elif isinstance(layer, torch.nn.Dropout):
        return {"p": layer.p, "inplace": layer.inplace}
    elif isinstance(layer, torch.nn.ReLU):
        return {"inplace": layer.inplace}
    return {}


def extract_hyperparams_from_layers(
    sequential_model: nn.Sequential, use_idx=False
) -> defaultdict:
    """
    Extrait les hyperparamètres de chaque couche d'un modèle Sequential.

    Args:
        sequential_model (nn.Sequential): Le modèle dont extraire les hyperparamètres.

    Returns:
        defaultdict: Un dictionnaire où les clés sont les noms de couches et les valeurs sont des dictionnaires d'hyperparamètres.
    """
    hyperparams = defaultdict(dict)

    for idx, layer in enumerate(sequential_model):
        if use_idx:
            layer_name = f"{layer.__class__.__name__}_{idx}"
        else:
            layer_name = f"{layer.__class__.__name__}"
        hyperparams[layer_name] = get_layer_hyperparams(layer)

    return hyperparams


def extract_hyperparams_from_layers_bis(
    sequential_model: nn.Sequential, use_idx=False
) -> defaultdict:
    """
    Extrait les hyperparamètres de chaque couche d'un modèle Sequential.

    Args:
        sequential_model (nn.Sequential): Le modèle dont extraire les hyperparamètres.

    Returns:
        defaultdict: Un dictionnaire où les clés sont les noms de couches et les valeurs sont des dictionnaires d'hyperparamètres.
    """
    hyperparams = []

    for idx, layer in enumerate(sequential_model):
        if use_idx:
            layer_name = f"{layer.__class__.__name__}_{idx}"
        else:
            layer_name = f"{layer.__class__.__name__}"
        hyperparams.append((layer_name, get_layer_hyperparams(layer)))

    return hyperparams


def get_checks_classifier(learner_model, dummy_flattened_size=128):

    with patch("torch.nn.Linear", wraps=nn.Linear) as mock_linear:
        learner_model._create_classifier(dummy_flattened_size)

    dict_info = {}
    dict_info["total_number_of_linear_layers"] = mock_linear.call_count

    for idx, call in enumerate(mock_linear.call_args_list):
        args, kwargs = call

        # Extraire in_features et out_features soit des args soit des kwargs
        in_features = kwargs.get("in_features") if "in_features" in kwargs else args[0]
        out_features = (
            kwargs.get("out_features") if "out_features" in kwargs else args[1]
        )

        dict_info[f"linear_layer_{idx+1}"] = {
            "in_features": in_features,
            "out_features": out_features,
        }

    learner_model.classifier = None
    return dict_info

    # endregion ==


# endregion ===

# region === Exercice 2 ===


def get_mock_fixed_trial(learner_search_space_func):
    # Créer un trial fixe avec des valeurs prédéterminées
    fixed_params = {
        "n_layers": 2,
        "n_filters_layer0": 16,
        "n_filters_layer1": 32,
        # "n_filters" : [16, 32],
        "kernel_size_layer0": 3,
        "kernel_size_layer1": 5,
        # "kernel_sizes": [3, 5],
        "dropout_rate": 0.001,
        "fc_size": 128,
        "learning_rate": 1e-3,
        "resolution": 32,
        "batch_size": 16,
    }

    trial = optuna.trial.FixedTrial(fixed_params)
    config = learner_search_space_func(trial)
    return config


def get_toy_trial(learner_search_space_func):
    def objective_toy(trial):
        params = learner_search_space_func(trial)
        return 0

    optuna.logging.disable_default_handler()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_toy, n_trials=1)
    trial = study.trials[0]
    return trial


def get_params_distributions(trial):

    keys = trial.params.keys()

    distributions_dict = {}

    for param in keys:
        # distributions_dict[param] = distribution_to_json(trial.distributions[param]) # décommenter pour générer les distributions attendues
        distributions_dict[param] = trial.distributions[param]
    return distributions_dict


def json_to_distribution(json_str):
    """Convertit une chaîne JSON en distribution Optuna."""
    import json
    data = json.loads(json_str)
    name = data["name"]
    attrs = data["attributes"]

    if name == "IntDistribution":
        return optuna.distributions.IntDistribution(
            low=attrs["low"],
            high=attrs["high"],
            step=attrs.get("step", 1),
            log=attrs.get("log", False)
        )
    elif name == "FloatDistribution":
        return optuna.distributions.FloatDistribution(
            low=attrs["low"],
            high=attrs["high"],
            step=attrs.get("step"),
            log=attrs.get("log", False)
        )
    elif name == "CategoricalDistribution":
        return optuna.distributions.CategoricalDistribution(choices=attrs["choices"])
    return None


def get_expected_params_distributions():
    expected_distributions_dict = {
        "n_layers": '{"name": "IntDistribution", "attributes": {"log": false, "step": 1, "low": 1, "high": 3}}',
        "n_filters_layer0": '{"name": "IntDistribution", "attributes": {"log": false, "step": 8, "low": 8, "high": 64}}',
        #'n_filters_layer1': '{"name": "IntDistribution", "attributes": {"log": false, "step": 8, "low": 8, "high": 64}}',
        "kernel_size_layer0": '{"name": "IntDistribution", "attributes": {"log": false, "step": 2, "low": 3, "high": 5}}',
        #'kernel_size_layer1': '{"name": "IntDistribution", "attributes": {"log": false, "step": 2, "low": 3, "high": 5}}',
        "dropout_rate": '{"name": "FloatDistribution", "attributes": {"step": null, "low": 0.1, "high": 0.5, "log": false}}',
        "fc_size": '{"name": "IntDistribution", "attributes": {"log": false, "step": 64, "low": 64, "high": 512}}',
        "learning_rate": '{"name": "FloatDistribution", "attributes": {"step": null, "low": 0.0001, "high": 0.01, "log": true}}',
        "resolution": '{"name": "CategoricalDistribution", "attributes": {"choices": [16, 32, 64]}}',
        "batch_size": '{"name": "CategoricalDistribution", "attributes": {"choices": [8, 16]}}',
    }

    # Convertir chaque valeur d'une chaîne JSON en dict Python
    # expected_distributions = {k: json_to_distribution(json.loads(v)) for k, v in expected_distributions_dict.items()}
    expected_distributions = {
        k: json_to_distribution(str(v)) for k, v in expected_distributions_dict.items()
    }

    return expected_distributions


# endregion ===


# region === Exercice 3 ===
def get_mock_params_ex3():
    mock_params_trial = optuna.trial.FixedTrial(
        {
            "n_layers": 2,
            "n_filters_layer0": 16,
            "n_filters_layer1": 32,
            "kernel_size_layer0": 3,
            "kernel_size_layer1": 3,
            "dropout_rate": 0.3,
            "fc_size": 128,
            "learning_rate": 0.001,
            "resolution": 32,
            "batch_size": 16,
        }
    )
    return mock_params_trial


def get_objects_during_trial(learner_obj_func, device, dataset_path):
    mock_trial = get_mock_params_ex3()

    # exécuter la fonction objet de l'apprenant sur des params fixes pour obtenir certains objets créés pendant le trial
    _ = learner_obj_func(
        trial=mock_trial,
        device=device,
        dataset_path=dataset_path,
        n_epochs=1,
        silent=True,
        test=True,
    )

    expected_parameters = mock_trial.params
    learner_objects = mock_trial.user_attrs

    transform = learner_objects["transform"]
    # train_loader = learner_objects['train_loader']
    model = learner_objects["model"]
    params_code = learner_objects["params_code"]
    return expected_parameters, transform, model, params_code


def remove_comments_ter(code):
    # Ce motif regex correspond aux commentaires dans le code
    pattern = r"#.*"

    # Utiliser re.sub() pour remplacer les commentaires par une chaîne vide
    code_without_comments = re.sub(pattern, "", code)

    # Diviser le code en lignes, supprimer chaque ligne et filtrer les lignes vides
    lines = code_without_comments.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]

    # Joindre les lignes non vides en une seule chaîne
    return "\n".join(non_empty_lines)


# endregion ===

# region === Exercice 4 ===
def check_calls_model_parameters(learner_func):
    # Configuration
    check_param1 = Mock()
    check_param1.requires_grad = True
    check_param1.numel.return_value = 100

    check_model = Mock()
    check_model.parameters.return_value = [check_param1]

    # Patcher torch.numel pour gérer les objets Mock
    with patch("torch.numel") as mock_torch_numel:

        def side_effect(param):
            if hasattr(param, "numel"):
                return param.numel()
            return 100  # valeur par défaut

        mock_torch_numel.side_effect = side_effect

        # --- Exécuter la fonction ---
        result = learner_func(check_model)

        # Vérifier que model.parameters() a été appelé
        check = check_model.parameters.called
    return check


def check_iterates_parameters(learner_func):
    # --- Configuration des mocks ---
    p1 = Mock()
    p1.requires_grad = True
    p1.numel.return_value = 100

    p3 = Mock()
    p3.requires_grad = True
    p3.numel.return_value = 300

    mock_model = Mock()
    mock_model.parameters.return_value = [p1, Mock(requires_grad=False), p3]

    # Patcher torch.numel pour gérer les objets Mock
    with patch("torch.numel") as mock_torch_numel:

        def side_effect(param):
            if hasattr(param, "numel"):
                return param.numel()
            return 100  # valeur par défaut

        mock_torch_numel.side_effect = side_effect

        # --- Exécuter la fonction ---
        learner_func(mock_model)

        # Vérifier si torch.numel a été appelé (puisque votre solution utilise torch.numel)
        # OU si param.numel() a été appelé (pour la rétrocompatibilité)
        check = mock_torch_numel.called or p1.numel.called or p3.numel.called

    return check


def check_requires_grad(learner_func):
    """
    Vérification numérique que la fonction additionne correctement uniquement les paramètres qui nécessitent des gradients.
    """

    # --- Configuration des mocks ---
    p1 = Mock()
    p1.requires_grad = True
    p1.numel.return_value = 100

    # Mock pour un paramètre non entraînable
    p2 = Mock()
    p2.requires_grad = False
    p2.numel.return_value = 9999

    p3 = Mock()
    p3.requires_grad = True
    p3.numel.return_value = 300

    mock_model = Mock()
    mock_model.parameters.return_value = [p1, p2, p3]

    expected_result = 100 + 300

    # Patcher torch.numel pour gérer les objets Mock
    with patch("torch.numel") as mock_torch_numel:

        def side_effect(param):
            if hasattr(param, "numel"):
                return param.numel()
            return 0  # valeur par défaut

        mock_torch_numel.side_effect = side_effect

        # --- Exécuter la fonction ---
        result = learner_func(mock_model)
        check = result == expected_result
    return check


def check_numel(learner_func):
    # --- Configuration des mocks ---
    p1 = Mock()
    p1.requires_grad = True
    p1.numel.return_value = 100

    p3 = Mock()
    p3.requires_grad = True
    p3.numel.return_value = 300

    mock_model = Mock()
    mock_model.parameters.return_value = [p1, Mock(requires_grad=False), p3]

    # Patcher torch.numel pour gérer les objets Mock
    with patch("torch.numel") as mock_torch_numel:

        def side_effect(param):
            if hasattr(param, "numel"):
                return param.numel()
            return 100  # valeur par défaut

        mock_torch_numel.side_effect = side_effect

        # --- Exécuter la fonction ---
        learner_func(mock_model)

        # Vérifier si torch.numel a été appelé (pour votre solution)
        # OU si param.numel() a été appelé (pour la rétrocompatibilité)
        check = mock_torch_numel.called or (p1.numel.called and p3.numel.called)

    return check
# endregion ===

def check_color_jitter(learner_transform, expected_brightness, expected_contrast):

    color_jitter_found = False
    found_correct_jitter = False
    found_brightness_val = None
    found_contrast_val = None

    for transform in learner_transform.transforms:
        if isinstance(transform, transforms.ColorJitter):
            color_jitter_found = True
            found_brightness = transform.brightness
            found_contrast = transform.contrast

            # Try to infer the input-style value from the range for brightness
            if isinstance(found_brightness, tuple):
                lower = found_brightness[0]
                upper = found_brightness[1]
                if lower >= 0 and abs(upper - 1) == abs(lower - 1):
                    found_brightness_val = abs(lower - 1)
                else:
                    found_brightness_val = f"({lower:.1f}, {upper:.1f})"
            else:
                found_brightness_val = found_brightness

            # Try to infer the input-style value from the range for contrast
            if isinstance(found_contrast, tuple):
                lower = found_contrast[0]
                upper = found_contrast[1]
                if lower >= 0 and abs(upper - 1) == abs(lower - 1):
                    found_contrast_val = abs(lower - 1)
                else:
                    found_contrast_val = f"({lower:.1f}, {upper:.1f})"
            else:
                found_contrast_val = found_contrast

            if found_brightness == (1 - expected_brightness, 1 + expected_brightness) and \
               found_contrast == (1 - expected_contrast, 1 + expected_contrast):
                found_correct_jitter = True
            break

    return color_jitter_found, found_correct_jitter, found_brightness_val, found_contrast_val


def check_shuffle(data_loader, should_shuffle):
    """
    Vérifie si un DataLoader mélange les données en observant l'ordre des labels.
    """
    first_iteration_labels = []
    for _, labels in data_loader:
        first_iteration_labels.extend(labels.tolist())

    second_iteration_labels = []
    for _, labels in data_loader:
        second_iteration_labels.extend(labels.tolist())

    if should_shuffle:
        return first_iteration_labels != second_iteration_labels
    else:
        return first_iteration_labels == second_iteration_labels


def remove_comments(code):
    # Ce pattern regex correspond aux commentaires dans le code
    pattern = r'#.*'

    # Utilise re.sub() pour remplacer les commentaires par une chaîne vide
    code_without_comments = re.sub(pattern, '', code)

    # Divise le code en lignes, nettoie chaque ligne, et filtre les lignes vides
    lines = code_without_comments.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]

    # Rejoint les lignes non vides en une seule chaîne
    return '\n'.join(non_empty_lines)


class MockImageFolder(Dataset):
    """
    Un dataset fictif qui se comporte comme ImageFolder à des fins de test.
    """
    def __init__(self, num_samples, img_size=(32, 32)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_array = np.full((self.img_size[0], self.img_size[1], 3), idx % 256, dtype=np.uint8)
        image = Image.fromarray(image_array)
        label = torch.tensor(idx % 2)

        if self.transform:
            image = self.transform(image)

        return image, label

def generate_mock_datasets(train_size=100, val_size=75):
    # Crée des instances de la classe au niveau du module
    mock_train_dataset = MockImageFolder(num_samples=train_size)
    mock_val_dataset = MockImageFolder(num_samples=val_size)
    return mock_train_dataset, mock_val_dataset