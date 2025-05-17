import argparse  # argument parser
from SiameseNN import *
from Dataset import *
from Logger import *  # logging functions


def main() -> None:
    """
    Main function to run the pipeline.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    _description = "Run the Siamese Neural Network Pipeline from a Stage."
    parser = argparse.ArgumentParser(description=_description)
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Stage to run (0, 1, ..., N). Default: 0.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to run. Default: 10.",
    )
    parser.add_argument(
        "--embedding",
        type=int,
        default=11,
        help="Model to run (0, 1, ..., N). Default: 11.",
    )

    args = parser.parse_args()
    print(args)
    run(args.stage, args.epochs, args.embedding)


def run(stage: int = None, EPOCHS: int = None, embedding: int = None) -> None:
    """
    Run the pipeline.

    Parameters
    ----------
        stage (int): Stage to run (0, 1, ..., N). Default: 0.
        EPOCHS (int): Number of epochs to run. Default: 10.
        embedding (int): Embedding to use (0, 1, ..., N). Default: 11.

    Returns
    -------
        None
    """

    if stage is None:
        stage = 3

    if EPOCHS is None:
        EPOCHS = 10

    if embedding is None:
        embedding = 11

    avbl_stgs = [i for i in range(0, 6)]  # Stages available to run
    try:
        msg_assert = "Stage {} is not available. Available stages are {}."
        assert stage in avbl_stgs, msg_assert.format(stage, avbl_stgs)
    except AssertionError as e:
        log_error(e)
        stage = avbl_stgs[-1]

    log_info("###########################################")
    log_info(f"Running from stage {stage} with {EPOCHS} epochs.")
    log_info("###########################################")
    constrain_gpu_memory()

    if stage <= 0:
        log_info("===========================================")
        log_info(f"Stage 0 - Dataset creation")
        log_info("===========================================")
        create_dataset()
    if stage <= 1:
        log_info("===========================================")
        log_info(f"Stage 1 - Make paired images")
        log_info("===========================================")
        create_siamese_datasets()
    if stage <= 2:
        log_info("===========================================")
        log_info(f"Stage 2 - Show paired images")
        log_info("===========================================")
        show_paired_images_info()
    if stage <= 3:
        log_info("===========================================")
        log_info(f"Stage 3 - Siamese NN")
        log_info("===========================================")
        create_siamese_model(EPOCHS=EPOCHS, embedding_type=embedding)
    if stage <= 4:
        log_info("===========================================")
        log_info(f"Stage 4 - Test - Training samples")
        log_info("===========================================")
        test_model(dataset="train", embedding_type=embedding)
    if stage <= 5:
        log_info("===========================================")
        log_info(f"Stage 5 - Test - Validation samples")
        log_info("===========================================")
        test_model(dataset="val", embedding_type=embedding)
    if stage <= 6:
        log_info("===========================================")
        log_info(f"Stage 6 - Test - Test samples")
        log_info("===========================================")
        test_model(embedding_type=embedding)
        exit(0)
    if stage <= 7:
        log_info("===========================================")
        log_info(f"Stage 7 - Test Symmetrical Datasets")
        log_info("===========================================")

        log_info("Loading siamese pre-trained model")
        siamese_nn = load_model_w_weights(embedding_type=embedding)

        import pandas as pd

        my_datasets = ["vow", "num", "jap", "kor"]
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

            idx = 0
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
                log_info(f"Probability Predicted for {idx}: {label_hat}")
                idx += 1
    if stage <= 8:
        log_info("===========================================")
        log_info(f"Stage 8 - TF Lite Optimization")
        log_info("===========================================")

        log_info("Loading siamese pre-trained model")
        siamese_nn = load_model_w_weights(embedding_type=embedding)


if __name__ == "__main__":
    main()
