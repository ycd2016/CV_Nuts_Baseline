import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


nlist = os.listdir("neg")
plist = os.listdir("pos")
x = np.zeros((len(nlist) + len(plist), 224, 224, 3))
for i, img in enumerate(nlist):
    x[i] = (
        np.asarray(
            Image.open(f"./neg/{img}")
            .convert("RGB")
            .resize((224, 224), Image.ANTIALIAS)
        )
        / 255.0
    )
for i, img in enumerate(plist):
    x[i + len(nlist)] = (
        np.asarray(
            Image.open(f"./pos/{img}")
            .convert("RGB")
            .resize((224, 224), Image.ANTIALIAS)
        )
        / 255.0
    )
y = np.zeros((len(nlist) + len(plist),))
y[len(nlist) :] = 1
ss = StandardScaler()
x = ss.fit_transform(x.reshape((-1, 3))).reshape((-1, 224, 224, 3))
with open("scaler.pkl", "wb") as f:
    pickle.dump(ss, f)


def Net():
    x = Input(shape=(224, 224, 3))
    y = EfficientNetB0(include_top=False)(x)
    y = GlobalMaxPooling2D()(y)
    y = Dense(1, "sigmoid")(y)
    return Model(x, y)


plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15)
early_stopping = EarlyStopping(monitor="val_loss", patience=45)
skf = StratifiedKFold(5, shuffle=True)
for f, (trn, val) in enumerate(skf.split(x, y)):
    checkpoint = ModelCheckpoint(f"fold{f}.h5", monitor="val_loss", save_best_only=True)
    model = Net()
    model.compile(optimizer=Adam(4e-4), loss="binary_crossentropy", metrics=["acc"])
    model.fit(
        x[trn],
        y[trn],
        batch_size=40,
        epochs=300,
        callbacks=[plateau, early_stopping, checkpoint],
        validation_data=(x[val], y[val]),
    )
    model = load_model(f"fold{f}.h5")
    model.save(f"fold{f}.h5", include_optimizer=False)
