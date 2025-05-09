import tensorflow as tf
import numpy as np
import random
import pandas as pd
import os
from typing import Tuple
import matplotlib.pyplot as plt
from Logger import *  # logging functions


def load_mnist_dataset(
    normalize: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and preprocess the MNIST dataset.

    Args:
        normalize (bool): Whether to normalize the input data.

    Returns:
        Tuple containing (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    log_info("MNIST successfully loaded")

    len_train = len(X_train)
    idx_val = np.random.choice(len_train, int(len_train * 0.1), replace=False)
    X_val, y_val = X_train[idx_val], y_train[idx_val]

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_val = X_val.reshape((X_val.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    if normalize:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0

    log_info(
        "MNIST successfully splitted into "
        + f"train ({X_train.shape[0]}), "
        + f"val ({X_val.shape[0]}), "
        + f"and test ({X_test.shape[0]})"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataset() -> None:
    """
    Create and save MNIST dataset splits as CSV files.

    This function loads the MNIST dataset, splits it into training, validation,
    and test sets, and saves each split as a CSV file in the "datasets" directory.

    The CSV files contain 785 columns: "label" and "pixel0" to "pixel783".

    Returns
    -------
        None
    """

    dataset_folders = ["datasets", "models"]
    for folder in dataset_folders:
        if not os.path.exists(folder):
            log_warning(f"Folder {folder} doesn't exist, creating it...")
            os.mkdir(folder)

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset()

    log_info("Creating csv dataset files")

    train_pixels = pd.DataFrame(X_train)
    train_pixels.columns = [f"pixel{i}" for i in range(train_pixels.shape[1])]
    train_pixels["label"] = y_train
    cols = ["label"] + [f"pixel{i}" for i in range(train_pixels.shape[1] - 1)]
    train = train_pixels[cols]
    train.to_csv("datasets/train.csv", index=False)
    log_success("datasets/train.csv successfully created")

    val_pixels = pd.DataFrame(X_val)
    val_pixels.columns = [f"pixel{i}" for i in range(val_pixels.shape[1])]
    val_pixels["label"] = y_val
    cols = ["label"] + [f"pixel{i}" for i in range(val_pixels.shape[1] - 1)]
    val = val_pixels[cols]
    val.to_csv("datasets/val.csv", index=False)
    log_success("datasets/val.csv successfully created")

    test_pixels = pd.DataFrame(X_test)
    test_pixels.columns = [f"pixel{i}" for i in range(test_pixels.shape[1])]
    test_pixels["label"] = y_test
    cols = ["label"] + [f"pixel{i}" for i in range(test_pixels.shape[1] - 1)]
    test = test_pixels[cols]
    test.to_csv("datasets/test.csv", index=False)
    log_success("datasets/test.csv successfully created")


def make_paired_indices(
    y: np.ndarray,
    num_pairs: int = 5000,
    balanced: bool = True,
    random_seed: int = 2025,
) -> pd.DataFrame:
    """
    Crea pares de índices para entrenamiento siamés con EXACTAMENTE num_pairs pares.

    Parameters
    ----------
        y: np.ndarray
            Etiquetas de las imágenes.
        num_pairs: int, optional
            Número de pares a crear (default: 100000).
        balanced: bool, optional
            Si se desea generar pares balanceados (default: True).
        random_seed: int, optional
            Semilla para la aleatoriedad (default: 2025).

    Returns
    -------
        DataFrame con columns: idx_1, idx_2, similarity
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    n = len(y)
    indices = np.arange(n)
    pairs = []

    # Set the indices for each class (e.g. label 0 has indices [0, 8, 13, ...])
    class_indices = {cls: indices[y == cls] for cls in np.unique(y)}
    classes = list(class_indices.keys())  # get the classes [0, 1, 2, ..., 9]

    if balanced:
        if num_pairs % 2 != 0:
            num_pairs += 1  # Ensure num_pairs is even
        num_pos = num_pairs // 2
        num_neg = num_pairs // 2
    else:
        num_pos = int(num_pairs * 0.5)  # Half of num_pairs (rounded down)
        num_neg = num_pairs - num_pos

    # Generate positive pairs
    pos_count = 0
    attempts = 0
    max_attempts = num_pos * 2  # limit to avoid inf loops

    # while is not the enoguh pairs or attempts are less than max attempts (2 times num of pairs)
    while pos_count < num_pos and attempts < max_attempts:
        cls = np.random.choice(classes)  # choose a random class
        same_class_idx = class_indices[cls]  # get the indices of that class
        if len(same_class_idx) >= 2:  # if there are at least 2 indices
            # Choose 2 random indices from the same class (ensure no same index)
            i1, i2 = np.random.choice(same_class_idx, 2, replace=False)
            pairs.append((i1, i2, 1))  # append as positive pair
            pos_count += 1  # increment positive pair count
        attempts += 1  # increment attempts

    if pos_count < num_pos:
        log_warning(f"Just {pos_count}/{num_pos} positive pairs were generated")

    # Generar pares negativos (diferente clase)
    neg_count = 0
    attempts = 0
    max_attempts = num_neg * 2

    while neg_count < num_neg and attempts < max_attempts:
        # Choose 2 random classes (ensure no same class)
        cls1, cls2 = np.random.choice(classes, 2, replace=False)
        # choose a random index from class 1
        idx1 = np.random.choice(class_indices[cls1])
        # choose a random index from class 2
        idx2 = np.random.choice(class_indices[cls2])
        pairs.append((idx1, idx2, 0))  # append as negative pair
        neg_count += 1  # increment negative pair count
        attempts += 1  # increment attempts

    if neg_count < num_neg:
        log_warning(f"Just {neg_count}/{num_neg} negative pairs were generated")

    random.shuffle(pairs)  # shuffle
    pairs = pairs[:num_pairs]  # keep only num_pairs

    # Check if pairs are balanced
    if balanced and len(pairs) > 0:
        similarity_counts = pd.Series([p[2] for p in pairs]).value_counts()
        if abs(similarity_counts.get(1, 0) - similarity_counts.get(0, 0)) > 1:
            log_warning("Pairs are not perfectly balanced")

    # Convert indices to DataFrame
    pairs_df = pd.DataFrame(pairs, columns=["idx_1", "idx_2", "similarity"])

    log_info(
        f"Generated {len(pairs_df)} pairs ({pairs_df['similarity'].sum()} similar, "
        + f"{len(pairs_df)-pairs_df['similarity'].sum()} different)"
    )

    return pairs_df


def create_siamese_datasets() -> None:
    """
    Crea datasets para red siamesa a partir de los CSV existentes.
    Guarda archivos con pares de índices para train, val y test.
    """
    log_info("Creating siamese datasets")

    # Cargar los datasets originales para obtener las etiquetas
    train_df = pd.read_csv("datasets/train.csv")
    val_df = pd.read_csv("datasets/val.csv")
    test_df = pd.read_csv("datasets/test.csv")

    # Crear pares para cada conjunto
    log_info("Generating training pairs")
    train_pairs = make_paired_indices(train_df["label"].values, num_pairs=30000)
    train_pairs.to_csv("datasets/siamese_train_pairs.csv", index=False)

    log_info("Generating validation pairs")
    val_pairs = make_paired_indices(val_df["label"].values, num_pairs=5000)
    val_pairs.to_csv("datasets/siamese_val_pairs.csv", index=False)

    log_info("Generating test pairs")
    test_pairs = make_paired_indices(test_df["label"].values, num_pairs=9000)
    test_pairs.to_csv("datasets/siamese_test_pairs.csv", index=False)

    log_info("Siamese pair datasets created successfully")


def plot_images(
    pairs_df: pd.DataFrame,
    original_df: pd.DataFrame,
    num_pairs: int = 5,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    # Seleccionar aleatoriamente algunos pares
    """
    Muestra pares de imágenes con sus etiquetas y similitud (similar/diferente).

    Parameters
    ----------
    pairs_df : pd.DataFrame
        DataFrame con columnas idx_1, idx_2, similarity que contiene los pares de índices
    original_df : pd.DataFrame
        DataFrame con las imágenes originales y sus etiquetas
    num_pairs : int, optional
        Número de pares a mostrar (default: 5)
    figsize : Tuple[int, int], optional
        Tamaño de la figura (default: (10, 5))

    Returns
    -------
    None
    """
    sample_pairs = pairs_df.sample(num_pairs)

    for _, row in sample_pairs.iterrows():
        # Obtener las imágenes y etiquetas
        img1 = original_df.iloc[row["idx_1"], 1:].values.reshape(28, 28)
        label1 = original_df.iloc[row["idx_1"], 0]

        img2 = original_df.iloc[row["idx_2"], 1:].values.reshape(28, 28)
        label2 = original_df.iloc[row["idx_2"], 0]

        similarity = row["similarity"]

        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Mostrar imágenes
        ax1.imshow(img1, cmap="gray")
        ax1.set_title(f"Imagen 1\nLabel: {label1}")
        ax1.axis("off")

        ax2.imshow(img2, cmap="gray")
        ax2.set_title(f"Imagen 2\nLabel: {label2}")
        ax2.axis("off")

        # Añadir supertítulo con la similitud
        similarity_text = "SIMILARES (1)" if similarity == 1 else "DIFERENTES (0)"
        fig.suptitle(
            f"Par {'similar' if similarity == 1 else 'no similar'}: {similarity_text}\n"
            f"Label1: {label1} - Label2: {label2}",
            fontsize=12,
        )

        plt.tight_layout()
        plt.show()


def separate_for_training(df: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_df = pd.read_csv(f"datasets/{df}.csv")
    test_df = test_df.drop("label", axis=1)
    test_pairs = pd.read_csv(f"datasets/siamese_{df}_pairs.csv")

    img1_t = test_df.iloc[test_pairs["idx_1"]].values.reshape(-1, 28, 28, 1)
    img2_t = test_df.iloc[test_pairs["idx_2"]].values.reshape(-1, 28, 28, 1)
    sim_t = test_pairs["similarity"].values

    return img1_t, img2_t, sim_t


def show_paired_images_info(verbose: bool = False) -> None:
    """
    Display information about paired images datasets.

    This function reads paired images information from CSV files for training,
    validation, and test datasets, and logs the total number of pairs, as well
    as the number of positive (similar) and negative (different) pairs for each
    dataset.

    Parameters
    ----------
    verbose : bool, optional
        A flag to indicate if additional information should be logged.
        (default: False)

    Returns
    -------
    None
    """

    type_dataset = ["train", "val", "test"]
    for t in type_dataset:
        pairs = pd.read_csv(f"datasets/siamese_{t}_pairs.csv")
        pos = pairs["similarity"].value_counts()[1]
        neg = pairs["similarity"].value_counts()[0]
        log_info(f"{t} has {pairs.shape[0]} imgs, pos: {pos} and neg: {neg}")
        if verbose:
            df = pd.read_csv(f"datasets/{t}.csv")
            plot_images(pairs, df, num_pairs=3)
