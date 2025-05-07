# -----------------------------------------------###############################
# -----------------------------------------------###############################
# This part is added until tensorflow 2.20 is released
import os
from colorama import Fore, Style
from time import strftime
import shutil

# import the List, Tuple, and Dict types
from typing import List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Suppress TensorFlow logging

import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    raise SystemError(
        f"{Fore.RED}GPU device not found{Style.RESET_ALL}"
    )  # raise exception if no GPU

print("\x1b[2J\x1b[H")  # clear screen after 'with tf.device('/device:GPU:0'):'
# -----------------------------------------------###############################
# -----------------------------------------------###############################

import numpy as np
import random
random.seed(2025)
np.random.seed(2025)
tf.random.set_seed(2025)

import matplotlib.pyplot as plt

from colorama import Fore, Style

from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Layer, Conv2D, Dense, AveragePooling2D, MaxPooling2D, Input, Flatten, Dropout  # type: ignore

import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename=f"{strftime('%Y-%m-%d')}.log",
    filemode="w",
    format="[%(levelname)-7s][%(asctime)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log_info(message):
    print(f"[{Fore.BLUE}  INFO {Style.RESET_ALL}][{strftime('%Y-%m-%d %H:%M:%S')}] - {message}{Style.RESET_ALL}")
    logging.info(message)

def log_error(message):
    print(f"[{Fore.RED} ERROR {Style.RESET_ALL}][{strftime('%Y-%m-%d %H:%M:%S')}] - {message}{Style.RESET_ALL}")
    logging.error(message)

def log_warning(message):
    print(f"[{Fore.YELLOW}WARNING{Style.RESET_ALL}][{strftime('%Y-%m-%d %H:%M:%S')}] - {message}{Style.RESET_ALL}")
    logging.warning(message)

def log_success(message):
    print(f"[{Fore.GREEN} DEBUG {Style.RESET_ALL}][{strftime('%Y-%m-%d %H:%M:%S')}] - {message}{Style.RESET_ALL}")
    logging.debug(message)


def progress_bar(
    count: int,
    total: int,
    suffix: str = "",
) -> None:
    """
    Display a progress bar.

    Parameters:
    -----------
    count : int
        The current count.
    total : int
        The total count.
    suffix : str, optional
        The suffix to display.

    Returns:
    --------
    None
    """
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = (
        f"{Fore.GREEN}#" * filled_len
        + f"{Fore.RED}_" * (bar_len - filled_len)
        + f"{Style.RESET_ALL}"
    )

    # if count == total-1 print the bar but without the \r
    if count == total - 1:
        print(
            f"[{Fore.GREEN} DEBUG {Style.RESET_ALL}][{strftime('%Y-%m-%d %H:%M:%S')}]"
            + "[%s] %s%s ...%s" % (bar, percents, "%", suffix)
        )
    else:
        print(
            f"[{Fore.BLUE}  INFO {Style.RESET_ALL}][{strftime('%Y-%m-%d %H:%M:%S')}]"
            + "[%s] %s%s ...%s\r" % (bar, percents, "%", suffix),
            end="",
        )


def constrain_gpu_memory() -> None:
    """
    Constrain GPU memory usage.

    Returns:
    --------
    None
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")  # get gpus
    if gpus is not None:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            log_info(
                f"{Fore.GREEN}{Style.BRIGHT}{len(gpus)}{Style.RESET_ALL} Physical GPUs - "
                + f"{Fore.GREEN}{Style.BRIGHT}{len(logical_gpus)}{Style.RESET_ALL} Logical GPUs"
            )
        except RuntimeError as e:
            log_error(e)  # Memory growth must be set before GPUs have been initialized
    else:
        log_warning("No GPUs found")


def remove_folder_contents(folder: str) -> None:
    """
    Remove all files in a folder.

    Parameters:
    -----------
    folder : str
        Path to the folder.

    Returns:
    --------
    None
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            log_error(f"Failed to delete {file_path}. Reason: {e}")

    log_warning(f"Folder {folder} is being cleared. Contents: empty")


def create_folders() -> None:
    """
    Create the positive, negative, and anchor folders.

    Returns:
    --------
    None
    """

    POS_PATH = os.path.join("data", "positive")
    NEG_PATH = os.path.join("data", "negative")
    ANCHOR_PATH = os.path.join("data", "anchor")
    DATASET_PATH = os.path.join("data", "dataset")

    if not os.path.exists(POS_PATH):
        os.makedirs(POS_PATH)
        log_info(f"Positive path: {POS_PATH}")

    if not os.path.exists(NEG_PATH):
        os.makedirs(NEG_PATH)
        log_info(f"Negative path: {NEG_PATH}")

    if not os.path.exists(ANCHOR_PATH):
        os.makedirs(ANCHOR_PATH)
        log_info(f"Anchor path: {ANCHOR_PATH}")

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        log_info(f"Dataset path: {DATASET_PATH}")

    try:
        assert os.path.exists(POS_PATH)
        assert os.path.exists(NEG_PATH)
        assert os.path.exists(ANCHOR_PATH)
        assert os.path.exists(DATASET_PATH)
        log_success("Images folders successfully created/checked")
    except AssertionError:
        log_error("Error creating folders for images")


def load_dataset():
    mnist = tf.keras.datasets.mnist
    log_success("MNIST successfully loaded")

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    log_info(
        "MNIST successfully splitted into "
        + f"train ({X_train.shape[0]}), "
        + f"and test ({X_test.shape[0]})"
    )

    # return X_train, y_train, X_test, y_test
    # join X_train and X_test
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    log_success("Dataset joint into X and y for further manipulation")
    return X, y


def create_imgs(
    X: np.ndarray,
    path: str,
    label: int,
    transformations=False,
    max_images: int = 2500,
) -> None:
    """
    Create positive, negative, and anchor images.

    Parameters:
    -----------
    X : np.ndarray
        Input images.
    path : str
        Path to save the images.
    label : int
        Label of the images.
    transformations : bool, optional
        Whether to apply transformations to the images, by default False.
    max_images : int, optional
        Maximum number of images to create, by default 100.

    Returns:
    --------
    None
    """
    # get a random subset of images
    random_images_idx = np.random.choice(len(X), max_images, replace=False)
    X = X[random_images_idx]

    if transformations:
        random_ops = np.random.randint(5, size=len(X))
        for i, (image, op_idx) in enumerate(zip(X, random_ops)):
            transformed = apply_transformation(image, op_idx)
            plt.imsave(f"{path}/{label}_{i}.png", transformed, cmap="gray")
            progress_bar(i, len(X), f"Saving label:{label} images")
    else:
        for i, image in enumerate(X):
            plt.imsave(f"{path}/{label}_{i}.png", image, cmap="gray")
            progress_bar(i, len(X), f"Saving label:{label} images")


def apply_transformation(image: np.ndarray, op_idx: int) -> np.ndarray:
    """
    Apply a symmetrical transformation to the image based on index.

    Parameters:
    -----------
    image : np.ndarray
        Input image.
    op_idx : int
        Index of the transformation (0 to 3).

    Returns:
    --------
    np.ndarray
        Transformed image.
    """
    operations = [
        lambda img: img,
        lambda img: np.fliplr(img),
        lambda img: np.rot90(img),
        lambda img: np.flipud(img),
        lambda img: np.rot90(img, 2),
    ]
    try:
        return operations[op_idx](image)
    except IndexError:
        log_error(f"Invalid transformation index: {op_idx}")
        return image  # Return original as fallback


def preprocess_image(image_filepath: str) -> np.ndarray:
    """
    Preprocess an image file.

    Parameters:
    -----------
    image_filepath : str
        Path to the image file.

    Returns:
    --------
    np.ndarray
        Preprocessed image.
    """
    byte_image = tf.io.read_file(image_filepath)
    image = tf.image.decode_png(byte_image, channels=1)
    image = tf.image.resize(image, (64, 64))
    image = tf.cast(image, tf.float32) / 255.0
    return image


def load_image_folder(folder_path: str) -> tf.data.Dataset:
    """
    Load and preprocess images from a folder into a TensorFlow dataset.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing images.

    Returns:
    --------
    tf.data.Dataset
        Dataset of preprocessed images.
    """
    log_info(f"Loading images from {folder_path}")

    if not os.path.exists(folder_path):
        log_error(f"Folder {folder_path} does not exist")
        return None

    try:
        image_files = sorted(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        log_success(f"Found {len(image_files)} valid images in {folder_path}")
    except Exception as e:
        log_error(f"Error listing files in {folder_path}. Reason: {e}")
        return None

    if len(image_files) == 0:
        log_error(f"No valid image files found in {folder_path}")
        return None

    try:
        dataset = tf.data.Dataset.from_tensor_slices(image_files)
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        log_success(f"Successfully created dataset from {folder_path}")
        return dataset
    except Exception as e:
        log_error(f"Failed to preprocess images from {folder_path}. Reason: {e}")
        return None


def joint_dataset(
    anchor_images: tf.data.Dataset,
    positive_images: tf.data.Dataset,
    negative_images: tf.data.Dataset,
) -> tf.data.Dataset:
    """
    Create a joint dataset of anchor, positive, and negative images.

    Parameters:
    -----------
    anchor_images : tf.data.Dataset
        Dataset of anchor images.
    positive_images : tf.data.Dataset
        Dataset of positive images.
    negative_images : tf.data.Dataset
        Dataset of negative images.

    Returns:
    --------
    tf.data.Dataset
        Joint dataset of anchor, positive, and negative images.
    """
    label_ones = tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor_images)))
    label_zero = tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor_images)))

    positives = tf.data.Dataset.zip((anchor_images, positive_images, label_ones))
    negatives = tf.data.Dataset.zip((anchor_images, negative_images, label_zero))

    dataset = positives.concatenate(negatives)
    log_success("Successfully created joint dataset")

    return dataset


def plot_distribution(y: np.ndarray) -> dict:
    """
    Create a distribution plot of the labels in the dataset.

    Parameters:
    -----------
    y : np.ndarray
        Array of labels.

    Returns:
    --------
    dict
        Dictionary of label counts.
    """
    labels, counts = np.unique(y, return_counts=True)
    plt.bar(labels, counts)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Label Distribution")
    plt.show()

    return dict(zip(labels, counts))





















def make_embedding() -> Model:
    """
    Create an embedding model.

    Returns:
    --------
    Model
        Embedding model.
    """

    # # inp = Input(shape=(100,100,1), name='input_image')
    # inp = Input(shape=(64,64,1), name='input_image')
    
    # # First block
    # c1 = Conv2D(64, (10,10), activation='relu')(inp)
    # # m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    # m1 = MaxPooling2D(pool_size=(2, 2), padding='same')(c1)
    
    # # Second block
    # c2 = Conv2D(128, (7,7), activation='relu')(m1)
    # # m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    # m2 = MaxPooling2D(pool_size=(2, 2), padding='same')(c2)
    
    # # Third block 
    # c3 = Conv2D(128, (4,4), activation='relu')(m2)
    # # m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    # m3 = MaxPooling2D(pool_size=(2, 2), padding='same')(c3)
    
    # # Final embedding block
    # c4 = Conv2D(256, (4,4), activation='relu')(m3)
    # # c4 = Dropout(0.2)(c4)
    # f1 = Flatten()(c4)
    # d1 = Dense(4096, activation='sigmoid')(f1)
    
    # return Model(inputs=inp, outputs=d1, name='LeNet5')


    inputs = Input(shape=(64, 64, 1))
    x = Conv2D(6, kernel_size=(5, 5), activation="relu")(inputs)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, kernel_size=(5, 5), activation="relu")(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation="relu")(x)
    x = Dense(84, activation="relu")(x)
    # outputs = Dense(10, activation="softmax")(x) # original removed
    outputs = x
    return Model(inputs, outputs, name="LeNet5")

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model(embedding): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(64, 64, 1))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(64, 64, 1))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


@tf.function
def train_step(batch, siamese_model, binary_cross_loss, opt):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # Return loss
    return loss

def train(data: tf.data.Dataset,
          EPOCHS: int,
          siamese_model: Model,
          binary_cross_loss: tf.keras.losses.Loss,
          opt: tf.keras.optimizers.Optimizer,
          checkpoint: tf.train.Checkpoint,
          checkpoint_prefix: str) -> None:
    """
    Trains the siamese network
    
    Args:
        data (tf.data.Dataset): The training data
        EPOCHS (int): The number of epochs to train for
        siamese_model (Model): The siamese model
        binary_cross_loss (tf.keras.losses.Loss): The binary cross entropy loss
        opt (tf.keras.optimizers.Optimizer): The optimizer to use
        checkpoint (tf.train.Checkpoint): The checkpoint to save
        checkpoint_prefix (str): The prefix to use for the checkpoint
    """
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch, siamese_model, binary_cross_loss, opt)
            progbar.update(idx+1)
        
        # Save checkpoints
        if epoch % 5 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)
