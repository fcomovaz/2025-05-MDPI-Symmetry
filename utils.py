import io
import sys
import os
import pandas as pd
from Logger import *  # logging functions


def log_function_output(function: callable = None) -> None:
    try:
        stream = io.StringIO()  # create StringIO object
        sys.stdout = stream  # and redirect stdout
        function()  # call the function
        sys.stdout = sys.__stdout__  # restore stdout

        # log the output
        for line in stream.getvalue().splitlines():
            log_info(line)
    except Exception as e:
        log_error(f"Error logging {function.__name__} output: {e}")


def create_results_file(results_file: str = "results_tests.csv") -> None:
    full_path = f"results/{results_file}"
    try:
        cols = [
            "embedding",
            "dataset",
            "roc",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "r2",
            "inference_time",
            "execution_time",
        ]
        df = pd.DataFrame(columns=cols)
        df.to_csv(full_path, index=False)
        log_info(f"Created {results_file}")
    except Exception as e:
        log_error(f"Error creating {results_file}: {e}")


def update_results_file(
    results_file: str = "results_tests.csv",
    embedding_type: int = 11,
    metrics: dict = {},
) -> None:
    full_path = f"results/{results_file}"
    if not os.path.exists(full_path):
        create_results_file(results_file)

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
    embed_name = embed_dict[embedding_type]

    df = pd.read_csv(full_path)  # read the file

    mask = (df["embedding"] == embed_name) & (df["dataset"] == metrics["dataset"][0])
    if mask.any():
        # Actualizar sólo las columnas métricas
        for col, vals in metrics.items():
            df.loc[mask, col] = vals[0]
    else:
        # Insertar fila nueva (fusionamos embedding + metrics)
        row = {"embedding": embed_name}
        row.update({col: vals[0] for col, vals in metrics.items()})
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # 5. Guardamos de vuelta
    df.to_csv(full_path, index=False)