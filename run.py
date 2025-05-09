import argparse  # argument parser
from pipelines import run  # run pipeline functions


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

    my_desc = "Run the Siamese Neural Network Pipeline from a Stage."
    parser = argparse.ArgumentParser(description=my_desc)
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Stage to run (0, 1, ..., N). Default: 0.",
    )

    args = parser.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()
