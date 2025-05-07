from utils import *

run_from_stage = 7
# ############################ STAGES ############################
# 0. Create or reset the folders where the images will be saved.
# 1. Load, create the dataset. This is the longest step because it
#    load the mnist dataset, and split it into X and y.
#    Create the data set with anchor, positive and negative images
#    but in each iteration (label) these folders are cleaned up to
#    avoid mixing data with other labels, but doing this doesn't
#    allow us to create a tensorflow dataset later bc the images
#    need to exist to create the dataset.
#    To avoid this issue with tensorflow, we recreate the images
#    in the next step to fill the folders with the labeled images
#    that we need to create the dataset. It seems as a non
#    deterministic process due to the random choice of the images,
#    however, it's solved by using same the seed for whole pipeline.
# 2. Create tensorflow datasets. Once the images are stored in the
#    correct folders, we can create the tensorflow datasets. We split
#    the dataset into train and test datasets. These datasets are
#    stored in data/train_dataset and data/test_dataset for their
#    future use and avoid recreating them.
# 3. Create and train the model. This is also a custom step because
#    the siamese model is created from scratch using the tensorflow
#    functional API. At the end, the model is saved as a h5 file.
# 4. Test the model. We test the model using the test dataset.
# 5. Separate submodel. From our original siamese model, we extract
#    the LeNet5 submodel and we save it as a h5 file. This is useful
#    due to the fact that we will reduce the model for its usage
#    in a microcontroller.
# ################################################################

if run_from_stage <= 0:
    log_info("===============================================")
    log_success("Coupling Point 0 - Reset folders")
    log_info("===============================================")
    create_folders()
    remove_folder_contents("data/positive")
    remove_folder_contents("data/negative")
    remove_folder_contents("data/anchor")

if run_from_stage <= 1:
    log_info("===============================================")
    log_success("Coupling Point 1 - Load dataset")
    log_info("===============================================")
    # X_train, y_train, X_test, y_test = load_dataset()
    X, y = load_dataset()
    num_img: int = 600

    log_info("-----------------------------------------------")
    log_success("Create images from labels")
    log_info("-----------------------------------------------")
    dataset = tf.data.Dataset.from_tensor_slices(([], [], []))
    # selected_label = 4
    for i in range(10):
        selected_label = i
        log_info("-----------------------------------------------")
        log_info(f"Selected label: {selected_label}")
        log_info("-----------------------------------------------")

        remove_folder_contents("data/positive")
        remove_folder_contents("data/negative")
        remove_folder_contents("data/anchor")

        log_info("Starting to save anchor images")
        selected_x = X[y == selected_label]
        create_imgs(selected_x, "data/anchor", selected_label, max_images=num_img)

        log_info("Starting to save positive images")
        selected_x = X[y == selected_label]
        create_imgs(
            selected_x,
            "data/positive",
            selected_label,
            transformations=True,
            max_images=num_img,
        )

        log_info("Starting to save negative images")
        selected_x = X[y != selected_label]
        create_imgs(selected_x, "data/negative", selected_label, max_images=num_img)

        log_info("-----------------------------------------------")
        log_success("Create tensorflow Datasets")
        log_info("-----------------------------------------------")
        positive_images = load_image_folder("data/positive")
        negative_images = load_image_folder("data/negative")
        anchor_images = load_image_folder("data/anchor")

        log_info("Concatenating datasets")
        _dataset = joint_dataset(anchor_images, positive_images, negative_images)
        dataset = dataset.concatenate(_dataset)

    # RECREATE IMAGES after the dataset
    for i in range(10):
        selected_label = i
        log_info("-----------------------------------------------")
        log_info(f"Replacing label: {selected_label}")
        log_info("-----------------------------------------------")

        log_info("Starting to save anchor images")
        selected_x = X[y == selected_label]
        create_imgs(selected_x, "data/anchor", selected_label, max_images=num_img)

        log_info("Starting to save positive images")
        selected_x = X[y == selected_label]
        create_imgs(
            selected_x,
            "data/positive",
            selected_label,
            transformations=True,
            max_images=num_img,
        )

        log_info("Starting to save negative images")
        selected_x = X[y != selected_label]
        create_imgs(selected_x, "data/negative", selected_label, max_images=num_img)


    
    # create a pipeline
    # for training do .take(size)
    # for val/testing do .skip(size)
    dataset = dataset.cache()   # stores the dataset in memory
    dataset = dataset.shuffle(buffer_size=1024) # shuffles the dataset

    tf.data.Dataset.save(dataset, "data/dataset")

    try:
        samples = dataset.take(40)
        samp = samples.as_numpy_iterator().next()
        plt.subplot(121)
        plt.imshow(samp[0], cmap="gray")
        plt.subplot(122)
        plt.imshow(samp[1], cmap="gray")
        plt.title(f"Label: {samp[2]}")
        plt.show()
    except:
        log_error("Could not plot images")

if run_from_stage <= 2:
    log_info("===============================================")
    log_success("Coupling Point 2 - Create train and test datasets")
    log_info("===============================================")
    loaded_dataset = tf.data.Dataset.load("data/dataset")
    BATCH_SIZE = 32
    train_dataset = loaded_dataset.take(int(len(loaded_dataset) * 0.8)) # takes the first 80%
    train_dataset = train_dataset.batch(BATCH_SIZE) # batches the dataset
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # prefetches the dataset

    test_dataset = loaded_dataset.skip(int(len(loaded_dataset) * 0.8)) # skips the first 80%
    test_dataset = test_dataset.batch(BATCH_SIZE) # batches the dataset
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE) # prefetches the dataset

    # save the datasets
    tf.data.Dataset.save(train_dataset, "data/train_dataset")
    tf.data.Dataset.save(test_dataset, "data/test_dataset")

    log_success("Datasets successfully created")
    # samp = loaded_dataset.as_numpy_iterator().next()
    # plt.subplot(121)
    # plt.imshow(samp[0], cmap="gray")
    # plt.subplot(122)
    # plt.imshow(samp[1], cmap="gray")
    # plt.title(f"Label: {samp[2]}")
    # plt.show()


if run_from_stage <= 3:
    log_info("===============================================")
    log_success("Coupling Point 3 - Create model")
    log_info("===============================================")
    train_data = tf.data.Dataset.load("data/train_dataset")
    
    embedding_model = make_embedding()

    l1 = L1Dist()

    model = make_siamese_model(embedding_model)
    log_success("Model successfully created")

    # model.summary()

    binary_cross_loss = tf.losses.BinaryCrossentropy()

    opt = tf.keras.optimizers.Adam(0.0001)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=model)
    log_info("Checkpoints created")

    EPOCHS = 50
    train(train_data, EPOCHS, model, binary_cross_loss, opt, checkpoint, checkpoint_prefix)
    log_success("Model successfully trained")
    model.save_weights("model_weights.weights.h5")
    log_success("Model weights successfully saved")

    # test(test_data, model)

if run_from_stage <= 4:
    log_info("===============================================")
    log_success("Coupling Point 4 - Test model")
    log_info("===============================================")
    model = make_siamese_model(make_embedding())

    model.load_weights("model_weights.weights.h5")
    test_data = tf.data.Dataset.load("data/test_dataset")

    from tensorflow.keras.metrics import Precision, Recall  # type: ignore

    test_input, test_val, y_true = test_data.as_numpy_iterator().next()
    y_hat = model.predict([test_input, test_val])

    recall = Recall()  # Creating a metric object
    recall.update_state(y_true, y_hat)  # Calculating the recall value
    recall = recall.result().numpy()  # Return Recall Result

    precision = Precision()  # Creating a metric object
    precision.update_state(y_true, y_hat)  # Calculating the precision value
    precision = precision.result().numpy()  # Return Precision Result

    log_info(f"Recall   : {recall}")
    log_info(f"Precision: {precision}")

    log_success("Model successfully tested")

    # Set plot size 
    # for xx in range(10):
    #     plt.figure(figsize=(10,8))

    #     # Set first subplot
    #     plt.subplot(1,2,1)
    #     plt.imshow(test_input[xx])

    #     # Set second subplot
    #     plt.subplot(1,2,2)
    #     plt.imshow(test_val[xx])

    #     plt.suptitle(f"Test Images for score {y_true[xx]} predicted as {y_hat[xx]}")

    #     # Renders cleanly
    #     plt.show()

if run_from_stage <= 5:
    log_info("===============================================")
    log_success("Coupling Point 5 - Separate submodel")
    log_info("===============================================")
    model = make_siamese_model(make_embedding())
    model.load_weights("model_weights.weights.h5")
    # model.summary()

    try:
        submodel = model.get_layer("LeNet5")
    except:
        log_error("LeNet5 layer not found")
        raise Exception("LeNet5 layer not found")

    # submodel.summary()

    log_success("Model successfully separated")

    test_data = tf.data.Dataset.load("data/test_dataset")
    test_input, test_val, y_true = test_data.as_numpy_iterator().next()

    lenet5 = make_embedding()
    lenet5.set_weights(submodel.get_weights())

    pred_subm = lenet5.predict(test_val)
    pred_orig = submodel.predict(test_val)

    # how similar are the outputs
    from sklearn.metrics import r2_score, mean_squared_error

    log_info(f"R^2 of submodel vs parent model: {r2_score(pred_subm, pred_orig)}")
    log_info(
        f"MSE of submodel vs parent model: {mean_squared_error(pred_subm, pred_orig)}"
    )


if run_from_stage <= 6:
    log_info("===============================================")
    log_success("Coupling Point 6 - Load Chinese dataset")
    log_info("===============================================")
    
    model = make_siamese_model(make_embedding())
    model.load_weights("model_weights.weights.h5")
    log_success("Model successfully loaded")

    import pandas as pd
    dataset_chinese = pd.read_csv("images_chinese_mnist.csv")
    y = dataset_chinese["label"].values
    X = dataset_chinese.drop(columns=["label"]).values
    X_r = X.reshape(-1, 64, 64, 1).squeeze()
    # predict using the X_r
    X_p = X_r.astype(np.float32) / 255
    y_hat = model.predict([X_r, X_r])

    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    y_r = np.ones(len(y_hat))
    # counts = plot_distribution(y_hat)
    log_info(f"R^2 of model vs ground truth: {r2_score(y_r, y_hat)}")
    log_info(f"MSE of model vs ground truth: {mean_squared_error(y_r, y_hat)}")


    # plot y_r vs y_hat
    # import matplotlib.pyplot as plt
    # plt.plot(y_hat, label="y_hat")
    # plt.plot(y_r, label="y_r")
    # plt.legend()
    # plt.show()

if run_from_stage <= 7:
    model = make_siamese_model(make_embedding())
    model.load_weights("model_weights.weights.h5")
    log_success("Model successfully loaded")

    import pandas as pd
    # df = pd.read_csv("gallery/vow/data.csv")
    # y = df["label"].values
    # X = df.drop(columns=["label"]).values
    # # X_r = X.reshape(-1, 64, 64, 1).squeeze() # my image gallery
    # X_r = X.reshape(-1, 64, 64, 1).astype(np.float32) / 255.0 # my image gallery
    # X_p = [X_r[0] for i in range(len(X_r))]

    # # predict using the X_r
    # # X_r = X_r.astype(np.float32) / 255.0
    # # X_p = X_p.astype(np.float32) / 255.0
    # y_hat = model.predict([X_r, X_r])


    df   = pd.read_csv("gallery/num/data.csv")
    y    = df["label"].values
    X    = df.drop(columns=["label"]).values
    X_r  = (X.reshape(-1,64,64,1).astype(np.float32) / 255.0)

    # 2) crea X_p repitiendo la primera muestra
    # for idx in range(len(X_r)):
    #     ref_img = X_r[idx:idx+1]  # shape (1,64,64,1)
    idx = 0
    for ref_img in X_r:
        # ref_img tiene shape (64,64,1)
        X_p = np.repeat(ref_img[None, ...], X_r.shape[0], axis=0)  # (N,64,64,1)
        # X_p     = np.repeat(ref_img, X_r.shape[0], axis=0)
        #—> ahora X_p.shape == X_r.shape == (N,64,64,1)

        # 3) predice
        y_hat = model.predict([X_p, X_r], batch_size=32)
        # y_hat = np.round(y_hat, 2)
        # from label choose the one with highest probability in y_hat
        # so get the highst index and convert to label
        idx_label = np.argmax(y_hat.flatten(), axis=0)
        label_hat = y[idx_label]


        # log_info(f'Probability Distros for {idx}: {y_hat.flatten()}') # [0.5050751  0.6964604  0.1303581  0.43605146 0.67199785]
        log_info(f'Probability Predicted {idx}: {label_hat}') # [0.5050751  0.6964604  0.1303581  0.43605146 0.67199785]
        idx += 1