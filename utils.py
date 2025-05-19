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
        if results_file == "results_tests.csv":
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
        else:
            cols = ["embedding", "vow", "num", "jap", "kor", "hand"]
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
    embed_name = embed_dict.get(embedding_type, str(embedding_type))

    if not os.path.exists(full_path):
        create_results_file(results_file)

    df = pd.read_csv(full_path)

    if results_file == "results_tests.csv":
        # 2) Carga y busca fila
        dataset = metrics.get("dataset", [None])[0]
        mask = (df["embedding"] == embed_name) & (df["dataset"] == dataset)

        if mask.any():
            # Actualiza columnas existentes
            for col, vals in metrics.items():
                df.loc[mask, col] = vals[0]
        else:
            # Inserta fila nueva
            row = {"embedding": embed_name}
            row.update({col: vals[0] for col, vals in metrics.items()})
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # -------------------------------
    # CASO 2: otro archivo → write-only
    # -------------------------------
    else:
        # 2) Carga y busca fila
        mask = df["embedding"] == embed_name

        if mask.any():
            # Actualiza columnas existentes
            for col, vals in metrics.items():
                df.loc[mask, col] = vals[0]
        else:
            # Inserta fila nueva
            row = {"embedding": embed_name}
            row.update({col: vals[0] for col, vals in metrics.items()})
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # 3) Guarda de vuelta
    df.to_csv(full_path, index=False)
