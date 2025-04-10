# =====================================
# 0. Functions definition
# =====================================
from datetime import datetime

def log(step_msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {step_msg}")

# =====================================
# 1. Set the path from Kaggle
# =====================================
import kagglehub
path = kagglehub.dataset_download("gpreda/chinese-mnist")
# print("Path to dataset files:", path)
log(f"1/5 - Path set for kaggle accesing {path}")

# =====================================
# 2. The dataset is in a weird format
#    what we're going to do is only
#    download the images to put them
#    together into an array.
# =====================================
import cv2
import pandas as pd
import numpy as np

csv = pd.read_csv(path+'/chinese_mnist.csv')  # read the documentation for the dataset
filename = csv[['suite_id', 'sample_id', 'code']].values  # needed columns to web scraping

images = [cv2.imread(f"{path}/data/data/input_{suite_id}_{sample_id}_{code}.jpg")
          for suite_id, sample_id, code in filename ]
labels = [ [x - 1] for x in csv['code'].values ]  # need to compensate to 0-14 (code start in 1)

images = np.array(images)  # images in numpy array
labels = np.array(labels)  # labels in numpy array
# print("Train images shape:", images.shape)  # verify shapes
# print("Train labels shape:", labels.shape)  # verify shapes

log("2/5 - Images and labels loaded")

# =====================================
# 3. Images are natively in RGB so,
#    need to convert it to gray 
# =====================================
images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images])
log("3/5 - Images converted to gray")

# =====================================
# 4. Lastly, need to joint the images
#    and the labels into a pandas
#    dataframe for a better ml usage.
# =====================================
imagenes_aplanadas = images_gray.reshape(images_gray.shape[0], -1) # image flattening 1-D (64x64 -> 4096)
labels_flat = labels.flatten() # label flattening 1-D ([] -> discrete array)

columnas_pixeles = [f'pixel{i}' for i in range(imagenes_aplanadas.shape[1])] # pixel columns
df = pd.DataFrame(imagenes_aplanadas, columns=columnas_pixeles)  # dataframe creation
df.insert(0, 'label', labels_flat)  # Insert the label column at the beggining
# print(df.head())  # verification print

df.to_csv('images_chinese_mnist.csv', index=False)  # save the dataframe
log("4/5 - Dataframe saved")

# =====================================
# 5. Optionally, you can download it
# =====================================
# from google.colab import files    # this is not working if you pull the file and run it (need fix)
# files.download('images_chinese_mnist.csv') # if you are using colab, this can be helpful
# log("5/5 - Dataframe downloaded")
log("5/5 - Please, download the file from colab cache files")
