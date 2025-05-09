import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_images(images: list, labels: list = None):
    fig, axes = plt.subplots(1, 5)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set(xticks=[], yticks=[])
        if labels.any():
            ax.set_title(labels[i])

    plt.show()


def convert_gallery_to_csv(FOLDER: str, SIZE: int):
    images_orig = [
        cv2.imread(FOLDER + "/" + file)
        for file in os.listdir(FOLDER)
        if file.endswith(".png")
    ]
    images_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images_orig]
    images_invr = [cv2.bitwise_not(image) for image in images_gray]
    images_resz = [cv2.resize(image, (SIZE, SIZE)) for image in images_invr]
    images_flat = [image.flatten() for image in images_resz]

    # create a dataframe with columns pixel0, pixel1, ..., pixelN, and label where label is the name of the image
    pixel_cols = [f"pixel{i}" for i in range(SIZE * SIZE)]
    label_cols = [i for i in range(len(images_flat))]
    df = pd.DataFrame(images_flat, columns=pixel_cols)
    df["label"] = label_cols
    df.to_csv(f"{FOLDER}/data.csv", index=False)


if __name__ == "__main__":
    FOLDER = "num"  # folder with images
    SIZE = 64      # size of images
    convert_gallery_to_csv(FOLDER, SIZE)

    df = pd.read_csv(f"{FOLDER}/data.csv")

    labels = df["label"].values
    images_flat = df.drop("label", axis=1).values  # images flatten
    images_arry = images_flat.reshape(-1, SIZE, SIZE)

    plot_images(images_arry, labels)
