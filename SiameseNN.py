import tensorflow as tf
import numpy as np
import random
import time
from colorama import Fore, Style

from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, AveragePooling2D  # type: ignore
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Input, Flatten, Dropout  # type: ignore
from tensorflow.keras.layers import DepthwiseConv2D, Lambda, Concatenate, ReLU  # type: ignore
from tensorflow.keras.regularizers import l1, l2, l1_l2  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

from utils import log_function_output, update_results_file
from Dataset import separate_for_training
from Logger import *  # logging functions

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

    Returns
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


def make_embedding(embedding_type: int = 11, mshape: tuple = (28, 28, 1)) -> Model:
    """
    Create an embedding model.

    Parameters
    ----------
    embedding_type (int): Embedding type to use (0, 1, ..., N). Default: 11.

    Returns
    --------
    Model
        Embedding model.
    """

    embed_dict = {
        0: "SiameseEmbedding",
        1: "LeNet5",
        2: "LeNet5Mod1",
        3: "LeNet5Mod2",
        4: "SiameseEmbeddingMod",
        5: "UltraMinimalEmbedding",
        6: "UltraMinimalEmbeddingMod",
        7: "DepthwiseOnlyEmbedding",
        8: "SqueezeNetLiteEmbedding",
        9: "UltraDepthwiseHybridLight1",
        10: "UltraDepthwiseHybridLight2",
        11: "UltraDepthwiseHybridLight3",
    }

    reg_l1_l2 = l1_l2(1e-7, 1e-7)

    try:
        log_info(f"Embedding type: {embedding_type} - {embed_dict[embedding_type]}")
    except KeyError:
        log_error(f"Unknown embedding type: {embedding_type}. Using default: 11")
        embedding_type = 11

    if embedding_type == 0:
        inp = Input(shape=mshape, name="input_image")
        # First block
        c1 = Conv2D(64, (10, 10), activation="relu", kernel_regularizer=reg_l1_l2)(inp)
        m1 = MaxPooling2D(pool_size=(2, 2), padding="same")(c1)
        # Second block
        c2 = Conv2D(128, (7, 7), activation="relu", kernel_regularizer=reg_l1_l2)(m1)
        m2 = MaxPooling2D(pool_size=(2, 2), padding="same")(c2)
        # Third block
        c3 = Conv2D(
            128,
            (4, 4),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(m2)
        m3 = MaxPooling2D(pool_size=(2, 2), padding="same")(c3)
        # Final embedding block
        c4 = Conv2D(
            256,
            (4, 4),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(m3)
        f1 = Flatten()(c4)
        f1 = Dropout(0.5)(f1)
        d1 = Dense(4096, activation="sigmoid", kernel_regularizer=reg_l1_l2)(f1)
        return Model(inputs=inp, outputs=d1, name="SiameseEmbedding")

    if embedding_type == 1:
        inputs = Input(shape=mshape, name="input_image")
        x = Conv2D(
            6,
            kernel_size=(5, 5),
            activation="relu",
            kernel_regularizer=reg_l1_l2,
        )(inputs)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            16,
            kernel_size=(5, 5),
            activation="relu",
            kernel_regularizer=reg_l1_l2,
        )(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(120, activation="relu", kernel_regularizer=reg_l1_l2)(x)
        x = Dense(84, activation="relu", kernel_regularizer=reg_l1_l2)(x)
        outputs = Dense(10, activation="softmax", kernel_regularizer=reg_l1_l2)(x)
        return Model(inputs, outputs, name="LeNet5")

    if embedding_type == 2:
        inputs = Input(shape=mshape, name="input_image")
        x = Conv2D(
            6,
            kernel_size=(5, 5),
            activation="relu",
            kernel_regularizer=reg_l1_l2,
        )(inputs)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            16,
            kernel_size=(5, 5),
            activation="relu",
            kernel_regularizer=reg_l1_l2,
        )(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(120, activation="relu", kernel_regularizer=reg_l1_l2)(x)
        x = Dense(84, activation="relu", kernel_regularizer=reg_l1_l2)(x)
        outputs = x
        return Model(inputs, outputs, name="LeNet5Mod1")

    if embedding_type == 3:
        inputs = Input(shape=mshape, name="input_image")
        x = Conv2D(
            6,
            kernel_size=(5, 5),
            activation="relu",
            kernel_regularizer=reg_l1_l2,
        )(inputs)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Conv2D(
            16,
            kernel_size=(5, 5),
            activation="relu",
            kernel_regularizer=reg_l1_l2,
        )(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(120, activation="relu", kernel_regularizer=reg_l1_l2)(x)
        x = Dense(84, activation="relu", kernel_regularizer=reg_l1_l2)(x)
        outputs = Dense(10, activation="sigmoid")(x)
        return Model(inputs, outputs, name="LeNet5Mod2")

    if embedding_type == 4:
        inp = Input(shape=mshape, name="input_image")
        # Bloque 1: conv estándar
        x = Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(inp)
        x = MaxPooling2D((2, 2), padding="same")(x)
        # Bloque 2: conv separable (depthwise+pointwise)
        x = SeparableConv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
            depthwise_regularizer=reg_l1_l2,
            pointwise_regularizer=reg_l1_l2,
        )(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        # Bloque 3: conv separable
        x = SeparableConv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
            depthwise_regularizer=reg_l1_l2,
            pointwise_regularizer=reg_l1_l2,
        )(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        # Pooling global para comprimir H×W → 1
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        # Capa densa final sin activación (lineal) para embedding
        outputs = Dense(
            64,
            activation=None,
            name="embedding_vector",
            kernel_regularizer=reg_l1_l2,
        )(x)
        return Model(inputs=inp, outputs=outputs, name="SiameseEmbeddingMod")

    if embedding_type == 5:
        inp = Input(shape=mshape, name="inp")
        # Depthwise conv para atrapar patrones locales
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(inp)
        # Pointwise conv para mezclar canales
        x = Conv2D(16, (1, 1), activation="relu", kernel_regularizer=reg_l1_l2)(x)
        # Segundo bloque minimal
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(x)
        x = Conv2D(32, (1, 1), activation="relu", kernel_regularizer=reg_l1_l2)(x)
        # Compresión final
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        # out = Conv2D(64, (1,1), activation=None)(x[...,None,None])
        # out = out[:,0,0,:]
        return Model(inp, x, name="UltraMinimalEmbedding")

    if embedding_type == 6:
        inp = Input(shape=mshape, name="inp")
        # Depthwise conv para atrapar patrones locales
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(inp)
        # Pointwise conv para mezclar canales
        x = Conv2D(16, (1, 1), activation="relu", kernel_regularizer=reg_l1_l2)(x)
        # Segundo bloque minimal
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(x)
        x = Conv2D(32, (1, 1), activation="relu", kernel_regularizer=reg_l1_l2)(x)
        # Compresión final
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Conv2D(
            64,
            (1, 1),
            activation=None,
            kernel_regularizer=reg_l1_l2,
        )(x[..., None, None])
        out = out[:, 0, 0, :]
        return Model(inp, out, name="UltraMinimalEmbeddingMod")

    if embedding_type == 7:
        inp = Input(shape=mshape, name="inp")
        # Tres bloques depthwise solamente
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(inp)
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(x)
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Dense(
            64,
            activation=None,
            name="embedding_vector",
            kernel_regularizer=reg_l1_l2,
        )(x)
        return Model(inp, out, name="DepthwiseOnlyEmbedding")

    if embedding_type == 8:
        inp = Input(shape=mshape, name="inp")
        # Fire module
        x = Conv2D(
            16,
            (1, 1),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(
            inp
        )  # squeeze
        e1 = Conv2D(
            32,
            (1, 1),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(
            x
        )  # expand 1x1
        e3 = Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(
            x
        )  # expand 3x3
        x = Concatenate()([e1, e3])
        # Otro fire módulo
        x = Conv2D(
            24,
            (1, 1),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(x)
        e1 = Conv2D(
            48,
            (1, 1),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(x)
        e3 = Conv2D(
            48,
            (3, 3),
            activation="relu",
            padding="same",
            kernel_regularizer=reg_l1_l2,
        )(x)
        x = Concatenate()([e1, e3])
        # Pool + embedding
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Conv2D(
            64,
            (1, 1),
            activation=None,
            kernel_regularizer=reg_l1_l2,
        )(x[..., None, None])
        out = out[:, 0, 0, :]
        return Model(inp, out, name="SqueezeNetLiteEmbedding")

    if embedding_type == 9:
        inp = Input(shape=mshape, name="inp")
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(inp)
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(x)
        x = Conv2D(
            24,
            (1, 1),
            activation="relu",
            kernel_regularizer=reg_l1_l2,
        )(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Dense(16, activation=None, name="embedding_vector")(x)
        return Model(inputs=inp, outputs=out, name="UltraDepthwiseHybridLight1")

    if embedding_type == 10:
        inp = Input(shape=mshape, name="inp")
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(inp)
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(x)
        x = Conv2D(
            24,
            (1, 1),
            activation="relu",
            kernel_regularizer=reg_l1_l2,
        )(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Dense(8, activation=None, name="embedding_vector")(x)
        return Model(inputs=inp, outputs=out, name="UltraDepthwiseHybridLight2")

    if embedding_type == 11:
        inp = Input((28, 28, 1), name="inp")
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(inp)
        x = DepthwiseConv2D(
            (3, 3),
            padding="same",
            activation="relu",
            depthwise_regularizer=reg_l1_l2,
        )(x)
        x = Conv2D(2, (1, 1), activation="relu", kernel_regularizer=reg_l1_l2)(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Dense(
            4,
            activation=None,
            name="embedding_vector",
            kernel_regularizer=reg_l1_l2,
        )(
            x
        )  # 2 make it worst
        return Model(inputs=inp, outputs=out, name="UltraDepthwiseHybridLight3")


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


def load_model_w_weights(
    model_path: str = "models/siamese.keras",
    embedding_type: int = 11,
) -> Model:
    """
    Load a pre-trained siamese neural network model with weights.

    Parameters
    ----------
    model_path : str, optional
        Path to the model weights file (default: "models/siamese.keras").
    embedding_type : int, optional
        Type of embedding to use (default: 11).

    Returns
    -------
    Model
        The siamese neural network model with loaded weights.
    """

    embedding = make_embedding(embedding_type)
    siamese_nn = make_siamese_model(embedding)
    try:
        siamese_nn.load_weights(model_path)
        return siamese_nn
    except:
        log_error(f"Error loading model - wrong embedding type")
        return None


def create_siamese_model(
    EPOCHS: int = 10, embedding_type: int = 11, verbose: bool = False
):
    log_info("Loading siamese train inputs (img1, img2, similarity)")
    img1_t, img2_t, sim_t = separate_for_training("train")

    log_info("Loading siamese val inputs (img1, img2, similarity)")
    img1_v, img2_v, sim_v = separate_for_training("val")

    if verbose:
        from matplotlib import pyplot as plt

        random_idx = np.random.randint(0, len(sim_t))
        img1 = img1_t.iloc[random_idx].values
        img2 = img2_t.iloc[random_idx].values
        sim = sim_t.iloc[random_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img1.reshape(28, 28), cmap="gray")
        ax2.imshow(img2.reshape(28, 28), cmap="gray")
        fig.suptitle(f"Similarity: {sim}")
        plt.show()

    log_info("Creating the model")
    embedding = make_embedding(embedding_type)
    siamese_nn = make_siamese_model(embedding)

    log_info("Compiling the model")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    _loss = tf.keras.losses.BinaryCrossentropy()
    met = ["accuracy"]
    siamese_nn.compile(optimizer=opt, loss=_loss, metrics=met)

    log_function_output(siamese_nn.summary)

    log_info("Training the model")
    esc = EarlyStopping(
        monitor="val_loss",  # Métrica a monitorear, en este caso, la pérdida de validación
        min_delta=1e-4,  # Umbral que debe superar la métrica para considerarse una mejora
        patience=3,  # Número de épocas sin mejora antes de detener la formación
        restore_best_weights=True,
    )
    siamese_nn.fit(
        [img1_t, img2_t],
        sim_t,
        epochs=EPOCHS,
        validation_data=([img1_v, img2_v], sim_v),
        callbacks=[esc],
    )

    log_info("Saving the model")
    siamese_nn.save("models/siamese.keras")


def test_model(
    dataset: str = "test",
    embedding_type: int = 11,
    verbose: bool = False,
) -> None:
    """
    Test the siamese model.

    Parameters
    ----------
    dataset : str, optional
        Dataset to test on (default: "test").
    embedding_type : int, optional
        Type of embedding to use (default: 11).
    verbose : bool, optional
        Whether to show the ROC curve (default: False).

    Returns
    -------
    None
    """
    log_info("Loading siamese pre-trained model")
    siamese_nn = load_model_w_weights(embedding_type=embedding_type)

    log_info("Loading siamese test inputs (img1, img2, similarity)")
    img1_t, img2_t, sim_t = separate_for_training(dataset)

    t1 = time.time()
    pred = siamese_nn.predict([img1_t, img2_t])
    t2 = time.time()
    pred_time = 1000 * (t2 - t1) / len(sim_t)  # time per prediction (ms)
    pred = pred.flatten()  # flatten the predictions
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))  # normalize
    pred = (pred >= 0.6).astype(int)  # decision threshold
    now = time.strftime("%Y-%m-%d-%H:%M:%S")
    log_info(f"Time per prediction: {pred_time} ms")
    # log_info(f"Real similarity:      {sim_t[0:8]}")
    # log_info(f"Predicted similarity: {pred[0:8]}")

    from sklearn.metrics import (
        roc_curve,
        auc,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
        r2_score,
    )

    fpr, tpr, _ = roc_curve(sim_t, pred)  # return fpr, tpr, thresholds
    # fpr = false positive rate, tpr = true positive rate, thresholds = threshold values
    roc_auc = auc(fpr, tpr)  # auc returns the area under the curve
    precision = precision_score(sim_t, pred)
    recall = recall_score(sim_t, pred)
    f1 = f1_score(sim_t, pred)
    accuracy = accuracy_score(sim_t, pred)
    r2 = r2_score(sim_t, pred)

    log_info("Results:")
    log_info(f"ROC AUC:   {roc_auc}")
    log_info(f"Precision: {precision}")
    log_info(f"Recall:    {recall}")
    log_info(f"F1 Score:  {f1}")
    log_info(f"Accuracy:  {accuracy}")
    log_info(f"R2 Score:  {r2}")

    results_dict = {
        "dataset": [dataset],
        "roc": [roc_auc],
        "precision": [precision],
        "recall": [recall],
        "f1": [f1],
        "accuracy": [accuracy],
        "r2": [r2],
        "inference_time": [pred_time],
        "execution_time": [now],
    }
    update_results_file(embedding_type=embedding_type, metrics=results_dict)
    log_info("Saving results to results/results.csv")

    if verbose:
        import matplotlib.pyplot as plt

        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()


def test_custom_datasets(embedding: int = None, my_datasets: list = None) -> None:
    """
    Test the siamese model on custom datasets.

    Parameters
    ----------
    embedding_type : int, optional
        Type of embedding to use (default: 11).

    Returns
    -------
    None
    """
    log_info("Loading siamese pre-trained model")
    siamese_nn = load_model_w_weights(embedding_type=embedding)

    import pandas as pd

    if my_datasets is None:
        my_datasets = ["vow", "num", "jap", "kor", "hand"]
    scores = []  # scores of each dataset (for dictionary)
    for d in my_datasets:
        log_info(f"Dataset: {d}")
        try:
            df = pd.read_csv(f"gallery/{d}/data.csv")
        except:
            log_warning("Collab detected. Using subfolder 2025-05-MDPI-Symmetry")
            df = pd.read_csv(f"2025-05-MDPI-Symmetry/gallery/{d}/data.csv")
        y = df["label"].values
        X = df.drop(columns=["label"]).values
        X_r = X.reshape(-1, 28, 28, 1).astype(np.float32)  # current image size
        # X_r = tf.image.resize(X_r, (28, 28)) # resize in case it is not 28x28

        idx = 0  # index to know which image is being processed
        idx_correct = 0  # index to know which prediction is correct
        for ref_img in X_r:
            X_p = np.repeat(ref_img[None, ...], X_r.shape[0], axis=0)
            # X_p     = np.repeat(ref_img, X_r.shape[0], axis=0)
            # —> ahora X_p.shape == X_r.shape == (N,64,64,1)

            # 3) predice
            y_hat = siamese_nn.predict([X_r, X_p], batch_size=32)
            A = y_hat - np.min(y_hat)
            B = np.max(y_hat) - np.min(y_hat)
            y_hat = A / B  # normalize
            # y_hat = (y_hat >= 0.4).astype(int)  # decision threshold
            # y_hat = np.round(y_hat, 2)
            # from label choose the one with highest probability in y_hat
            # so get the highst index and convert to label
            idx_label = np.argmax(y_hat.flatten(), axis=0)
            label_hat = y[idx_label]

            # log_info(f"Probability Distribut for {idx}: {y_hat.flatten()}")
            idx_correct += int(idx == idx_label)  # check if prediction is correct
            log_info(f"Probability Predicted for {idx}: {label_hat}")
            idx += 1
        acc_percent = idx_correct / len(y) * 100
        scores.append(acc_percent)  # append accuracy 4 each dataset
        log_info(f"Accuracy for {d}: {acc_percent:.2f}%")
    avg_acc = sum(scores) / len(my_datasets)  # average accuracy
    scores.append(avg_acc)  # append average accuracy for all datasets
    my_datasets.append("overall")  # add an overall accuracy
    acc_dict = {k: [v] for k, v in zip(my_datasets, scores)}
    update_results_file(
        results_file="results_nonseen.csv",
        embedding_type=embedding,
        metrics=acc_dict,
    )
