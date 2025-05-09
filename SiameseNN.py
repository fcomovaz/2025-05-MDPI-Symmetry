import tensorflow as tf
from Logger import *  # logging functions
import numpy as np
import random
from colorama import Fore, Style

from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, AveragePooling2D  # type: ignore
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Input, Flatten, Dropout  # type: ignore
from tensorflow.keras.layers import DepthwiseConv2D, Lambda, Concatenate, ReLU  # type: ignore
from tensorflow.keras.regularizers import l1, l2, l1_l2  # type: ignore

# ---------------------------------------
# CHECK GPU
# ---------------------------------------
device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    log_error("GPU device not found.")
# ---------------------------------------

# ---------------------------------------
# SEEDING
# ---------------------------------------
random.seed(2025)
np.random.seed(2025)
tf.random.set_seed(2025)
# ---------------------------------------


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
                f"{len(gpus)} Physical GPUs - " + f"{len(logical_gpus)} Logical GPUs"
            )
        except RuntimeError as e:
            log_error(e)  # Memory growth must be set before GPUs have been initialized
    else:
        log_warning("No GPUs found")


def make_embedding() -> Model:
    """
    Create an embedding model.

    Returns:
    --------
    Model
        Embedding model.
    """

    # # 10 epochs - 100 imgs - 2M parameters - 8MB - 0.75 - 0.85 - NO - (9MB->2.3MB)
    # inp = Input(shape=(64,64,1), name='input_image')
    # # First block
    # c1 = Conv2D(64, (10,10), activation='relu')(inp)
    # m1 = MaxPooling2D(pool_size=(2, 2), padding='same')(c1)
    # # Second block
    # c2 = Conv2D(128, (7,7), activation='relu')(m1)
    # m2 = MaxPooling2D(pool_size=(2, 2), padding='same')(c2)
    # # Third block
    # c3 = Conv2D(128, (4,4), activation='relu')(m2)
    # m3 = MaxPooling2D(pool_size=(2, 2), padding='same')(c3)
    # # Final embedding block
    # c4 = Conv2D(256, (4,4), activation='relu')(m3)
    # f1 = Flatten()(c4)
    # d1 = Dense(4096, activation='sigmoid')(f1)
    # return Model(inputs=inp, outputs=d1, name='SiameseEmbedding')

    # # 10 epochs - 100 imgs - 337k parameters - 1.3MB - 0.81 - 0.76 - NO - (1.3MB->340kB)
    # inputs = Input(shape=(64, 64, 1), name="input_image")
    # x = Conv2D(6, kernel_size=(5, 5), activation="relu")(inputs)
    # x = AveragePooling2D(pool_size=(2, 2))(x)
    # x = Conv2D(16, kernel_size=(5, 5), activation="relu")(x)
    # x = AveragePooling2D(pool_size=(2, 2))(x)
    # x = Flatten()(x)
    # x = Dense(120, activation="relu")(x)
    # x = Dense(84, activation="relu")(x)
    # # outputs = Dense(10, activation="softmax")(x) # original removed
    # outputs = x
    # return Model(inputs, outputs, name="LeNet5")

    # # 10 epochs - 100 imgs - 19k parameters - 78kB - 0.12 - 0.66 - YES - (88kB->36kB)
    # inp = Input(shape=(64,64,1), name='input_image')
    # # Bloque 1: conv estándar
    # x = Conv2D(32, (3,3), activation="relu", padding="same", kernel_regularizer=l1_l2(0.01))(inp)
    # x = MaxPooling2D((2,2), padding="same")(x)
    # # Bloque 2: conv separable (depthwise+pointwise)
    # x = SeparableConv2D(64, (3,3), activation="relu", padding="same", depthwise_regularizer=l1_l2(0.01), pointwise_regularizer=l2(0.01))(x)
    # x = MaxPooling2D((2,2), padding="same")(x)
    # # Bloque 3: conv separable
    # x = SeparableConv2D(128, (3,3), activation="relu", padding="same", depthwise_regularizer=l1_l2(0.01), pointwise_regularizer=l2(0.01))(x)
    # x = MaxPooling2D((2,2), padding="same")(x)
    # # Pooling global para comprimir H×W → 1
    # x = GlobalAveragePooling2D()(x)
    # # Capa densa final sin activación (lineal) para embedding
    # outputs = Dense(64, activation=None, name="embedding_vector", kernel_regularizer=l1_l2(0.01))(x)
    # return Model(inputs=inp, outputs=outputs, name="SiameseEMbeddingMod")

    # # 10 epochs - 100 imgs - 779 parameters - 3kB - 1 - 0.5 - NO - (9kB)
    # inp = Input((64,64,1), name="inp")
    # # Depthwise conv para atrapar patrones locales
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(inp)
    # # Pointwise conv para mezclar canales
    # x = Conv2D(16, (1,1), activation="relu")(x)
    # # Segundo bloque minimal
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(x)
    # x = Conv2D(32, (1,1), activation="relu")(x)
    # # Compresión final
    # x = GlobalAveragePooling2D()(x)
    # # out = Conv2D(64, (1,1), activation=None)(x[...,None,None])
    # # out = out[:,0,0,:]
    # return Model(inp, x, name="UltraMinimalEmbedding")

    # # 10 epochs - 100 imgs - 939 parameters - 3.6kB - 1 - 0.5 - YES - (12kB)
    # inp = Input((64,64,1), name="inp")
    # # Depthwise conv para atrapar patrones locales
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(inp)
    # # Pointwise conv para mezclar canales
    # x = Conv2D(16, (1,1), activation="relu")(x)
    # # Segundo bloque minimal
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(x)
    # x = Conv2D(32, (1,1), activation="relu")(x)
    # # Compresión final
    # x = GlobalAveragePooling2D()(x)
    # out = Conv2D(64, (1,1), activation=None)(x[...,None,None])
    # out = out[:,0,0,:]
    # return Model(inp, out, name="UltraMinimalEmbeddingMod")

    # # 10 epochs - 100 imgs - 223 parameters - 892B - 1 - 0.5 - NO - (6kB)
    # inp = Input((64,64,1), name="inp")
    # # Tres bloques depthwise solamente
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(inp)
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(x)
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(x)
    # x = GlobalAveragePooling2D()(x)
    # out = Dense(64, activation=None, name="embedding_vector")(x)
    # return Model(inp, out, name="DepthwiseOnlyEmbedding")

    # # 10 epochs - 100 imgs - 18k parameters - 72kB - 0.06 - 0.5 - NO - (85kB->34kB)
    # inp = Input((64,64,1), name="inp")
    # # Fire module
    # x = Conv2D(16, (1,1), activation="relu", padding="same")(inp)  # squeeze
    # e1 = Conv2D(32, (1,1), activation="relu", padding="same")(x)  # expand 1x1
    # e3 = Conv2D(32, (3,3), activation="relu", padding="same")(x)  # expand 3x3
    # x = Concatenate()([e1, e3])
    # # Otro fire módulo
    # x = Conv2D(24, (1,1), activation="relu", padding="same")(x)
    # e1 = Conv2D(48, (1,1), activation="relu", padding="same")(x)
    # e3 = Conv2D(48, (3,3), activation="relu", padding="same")(x)
    # x = Concatenate()([e1, e3])
    # # Pool + embedding
    # x = GlobalAveragePooling2D()(x)
    # out = Conv2D(64, (1,1), activation=None)(x[...,None,None])
    # out = out[:,0,0,:]
    # return Model(inp, out, name="SqueezeNetLiteEmbedding")

    # # 10 epochs - 100 imgs - 485 parameters - 1.9kB - 1 - 0.5 - YES - (8kB)
    # inp = Input((64, 64, 1), name="inp")
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(inp)
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(x)
    # x = Conv2D(24, (1,1), activation="relu")(x)
    # x = GlobalAveragePooling2D()(x)
    # out = Dense(16, activation=None, name="embedding_vector")(x)
    # return Model(inputs=inp, outputs=out, name="UltraDepthwiseHybridLight")

    # # 10 epochs - 100 imgs - 277 parameters - 1kB - 1 - 0.5 - YES - (7kB)
    # inp = Input((64, 64, 1), name="inp")
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(inp)
    # x = DepthwiseConv2D((3,3), padding="same", activation="relu")(x)
    # x = Conv2D(24, (1,1), activation="relu")(x)
    # x = GlobalAveragePooling2D()(x)
    # out = Dense(8, activation=None, name="embedding_vector")(x)
    # return Model(inputs=inp, outputs=out, name="UltraDepthwiseHybridLight")

    inp = Input((28, 28, 1), name="inp")
    x = DepthwiseConv2D((3, 3), padding="same", activation="relu")(inp)
    x = DepthwiseConv2D((3, 3), padding="same", activation="relu")(x)
    x = Conv2D(2, (1, 1), activation="relu")(x)
    x = GlobalAveragePooling2D()(x)
    out = Dense(4, activation=None, name="embedding_vector")(x)  # 2 make it worst
    return Model(inputs=inp, outputs=out, name="UltraDepthwiseHybridLight")


def make_siamese_model(embedding):

    # Anchor image input in the network
    inp_img = Input(name="input_img", shape=(28, 28, 1))

    # Validation image in the network
    val_img = Input(name="validation_img", shape=(28, 28, 1))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = "distance"
    distances = siamese_layer(embedding(inp_img), embedding(val_img))

    # Classification layer
    clf = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[inp_img, val_img], outputs=clf, name="SiameseNN")


class L1Dist(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        """
        Init method for L1Dist layer.

        Parameters
        ----------
        **kwargs
            Keyword arguments for the parent class (Layer).
        """
        super().__init__()

    # Magic happens here - similarity calculation
    def call(
        self,
        input_embedding: tf.Tensor,
        validation_embedding: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the L1 distance between two embedding vectors.

        Args:
            input_embedding (tf.Tensor): The first embedding vector.
            validation_embedding (tf.Tensor): The second embedding vector.

        Returns:
            tf.Tensor: The L1 distance between the input and validation embeddings.
        """
        return tf.math.abs(input_embedding - validation_embedding)


def load_model_w_weights(model_path: str = "models/siamese.keras") -> Model:
    """
    Load a pre-trained siamese neural network model with weights.

    Parameters
    ----------
    model_path : str, optional
        Path to the model weights file (default: "models/siamese.keras").

    Returns
    -------
    Model
        The siamese neural network model with loaded weights.
    """

    embedding = make_embedding()
    siamese_nn = make_siamese_model(embedding)
    siamese_nn.load_weights(model_path)
    return siamese_nn
