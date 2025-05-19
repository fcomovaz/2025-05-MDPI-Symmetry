import argparse  # argument parser
import os
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
    parser.add_argument(
        "--cleanlogs",
        type=int,
        default=0,
        help="Clean logs folder. Default: 0.",
    )

    args = parser.parse_args()  # Parse the arguments
    if args.cleanlogs: os.system("rm logs/*.log")  # Clean logs
    log_config(args.embedding)  # Configure logging
    run(args.stage, args.epochs, args.embedding)  # Run the pipeline


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

    avbl_stgs = [i for i in range(0, 9)]  # Stages available to run
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
    if stage <= 7:
        log_info("===========================================")
        log_info(f"Stage 7 - Test Custom Datasets")
        log_info("===========================================")
        test_custom_datasets(embedding=embedding)
    # if stage <= 8:
    #     log_info("===========================================")
    #     log_info(f"Stage 8 - TF Lite Optimization")
    #     log_info("===========================================")

    #     log_info("Loading siamese pre-trained model")
    #     siamese_nn = load_model_w_weights(embedding_type=embedding)


if __name__ == "__main__":
    main()
