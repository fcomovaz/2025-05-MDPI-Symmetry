from SiameseNN import *
from Dataset import *
from Logger import *  # logging functions


def run(stage: int = None, EPOCHS: int = None) -> None:
    """
    Run the pipeline.

    Parameters
    ----------
        stage (int): Stage to run (0, 1, ..., N). Default: 0.

    Returns
    -------
        None
    """

    if stage is None:
        stage = 3

    if EPOCHS is None:
        EPOCHS = 10


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


        log_info("Loading siamese train inputs (img1, img2, similarity)")
        img1_t, img2_t, sim_t = separate_for_training("train")

        log_info("Loading siamese val inputs (img1, img2, similarity)")
        img1_v, img2_v, sim_v = separate_for_training("val")

        # from matplotlib import pyplot as plt
        # random_idx = np.random.randint(0, len(sim_t))
        # img1 = img1_t.iloc[random_idx].values
        # img2 = img2_t.iloc[random_idx].values
        # sim = sim_t.iloc[random_idx]

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img1.reshape(28, 28), cmap="gray")
        # ax2.imshow(img2.reshape(28, 28), cmap="gray")
        # fig.suptitle(f"Similarity: {sim}")
        # plt.show()

        log_info("Creating the model")
        embedding = make_embedding()
        siamese_nn = make_siamese_model(embedding)

        log_info("Compiling the model")
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        _loss = tf.keras.losses.BinaryCrossentropy()
        siamese_nn.compile(optimizer=opt, loss=_loss, metrics=["accuracy"])

        siamese_nn.summary()

        log_info("Training the model")
        siamese_nn.fit(
            [img1_t, img2_t],
            sim_t,
            epochs=EPOCHS,
            validation_data=([img1_v, img2_v], sim_v),
        )

        log_info("Saving the model")
        siamese_nn.save("models/siamese.keras")
    if stage <= 4:
        log_info("===========================================")
        log_info(f"Stage 4 - Test")
        log_info("===========================================")
        
        log_info("Loading siamese pre-trained model")
        siamese_nn = load_model_w_weights()
        
        log_info("Loading siamese test inputs (img1, img2, similarity)")
        img1_t, img2_t, sim_t = separate_for_training("test")

        pred = siamese_nn.predict([img1_t, img2_t])
        # normalize between 0 and 1
        # pred_max = np.max(pred)
        # pred_min = np.min(pred)
        # pred_n = (pred - pred_min) / (pred_max - pred_min)
        
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(sim_t, pred)
        roc_auc = auc(fpr, tpr)
        
        # plt.figure()
        # lw = 2
        # plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
        # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title("Receiver operating characteristic example")
        # plt.legend(loc="lower right")
        # plt.show()
    if stage <= 5:
        log_info("===========================================")
        log_info(f"Stage 5 - Test Symmetrical Datasets")
        log_info("===========================================")

        log_info("Loading siamese pre-trained model")
        siamese_nn = load_model_w_weights()

        import pandas as pd

        my_datasets = ['vow', 'num', 'jap', 'kor']
        for d in my_datasets:
            log_info(f"-----------------------------------------------")
            log_info(f"Dataset: {d}")
            log_info(f"-----------------------------------------------")
            try:
                df = pd.read_csv(f"gallery/{d}/data.csv")
            except:
                log_warning("Collab env detected. Using subfolder 2025-05-MDPI-Symmetry")
                df = pd.read_csv(f"2025-05-MDPI-Symmetry/gallery/{d}/data.csv")
            y    = df["label"].values
            X    = df.drop(columns=["label"]).values
            X_r  = (X.reshape(-1,64,64,1).astype(np.float32))
            X_r = tf.image.resize(X_r, (28, 28))

            idx = 0
            for ref_img in X_r:
                # ref_img tiene shape (64,64,1)
                X_p = np.repeat(ref_img[None, ...], X_r.shape[0], axis=0)  # (N,64,64,1)
                # X_p     = np.repeat(ref_img, X_r.shape[0], axis=0)
                #—> ahora X_p.shape == X_r.shape == (N,64,64,1)

                # 3) predice
                y_hat = siamese_nn.predict([X_r, X_p], batch_size=32)
                # y_hat = np.round(y_hat, 2)
                # from label choose the one with highest probability in y_hat
                # so get the highst index and convert to label
                idx_label = np.argmax(y_hat.flatten(), axis=0)
                label_hat = y[idx_label]


                log_info(f'Probability Distribut for {idx}: {y_hat.flatten()}') 
                log_info(f'Probability Predicted for {idx}: {label_hat}') 
                idx += 1
    if stage <= 6:
        log_info("===========================================")
        log_info(f"Stage 6 - TF Lite Optimization")
        log_info("===========================================")

        log_info("Loading siamese pre-trained model")
        siamese_nn = load_model_w_weights()

if __name__ == "__main__":
    run()
    log_info("Pipeline Done")
