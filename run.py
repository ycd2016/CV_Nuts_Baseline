import os
import sys
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


if __name__ == "__main__":
    path_cur = os.path.dirname(sys.argv[0])
    path_img = sys.argv[1]
    path_sub = sys.argv[2]
    ilist = os.listdir(path_img)
    x = np.zeros((len(ilist), 224, 224, 3))
    for i, png in enumerate(ilist):
        x[i] = (
            np.asarray(
                Image.open(os.path.join(path_img, png))
                .convert("RGB")
                .resize((224, 224), Image.ANTIALIAS)
            )
            / 255.0
        )
    with open(os.path.join(path_cur, "scaler.pkl"), "rb") as f:
        ss = pickle.load(f)
    x = ss.transform(x.reshape((-1, 3))).reshape((-1, 224, 224, 3))
    res = ["id,label"]
    classes = ["neg", "pos"]
    pred = np.zeros((len(ilist),))
    for i in range(5):
        model = load_model(os.path.join(path_cur, f"fold{i}.h5"))
        pred += model.predict(x).ravel() / 5.0
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    pred = pred.astype(int)
    for i, png in enumerate(ilist):
        res.append(png.split(".")[0] + "," + classes[pred[i]])
    with open(path_sub, "w") as f:
        f.write("\n".join(res) + "\n")
