import os
import sys
import numpy as np
import pandas as pd
from model import *
from PIL import Image
import tensorflow as tf
from multiprocessing import Pool
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def readimg(png):
    return np.asarray(
        Image.open(os.path.join(path_img, png))
        .resize((224, 224), Image.ANTIALIAS)
        .convert("RGB")
    )


def Net():
    x = Input((224, 224, 3))
    # 数据增广推理时自动跳过
    y = experimental.preprocessing.RandomFlip()(x)
    y = experimental.preprocessing.RandomTranslation(0.2, 0.2)(y)
    y = experimental.preprocessing.RandomRotation(0.2)(y)
    y = experimental.preprocessing.RandomZoom(0.2)(y)
    # 骨干网络，详见model.py
    y = create_RepVGG_tiny(1, True)(y)
    # 正样本的概率
    y = tf.math.sigmoid(y)
    return Model(x, y)


def loaddat():
    pool = Pool()
    # 第1步，读取图片
    dat = np.asarray(pool.map(readimg, ilist))
    pool.close()
    pool.join()
    # 第2步，进行标准化
    dat[:, :, :, 0] = (dat[:, :, :, 0] - 128.78203439) / 58.11709600
    dat[:, :, :, 1] = (dat[:, :, :, 1] - 120.90287413) / 56.96523662
    dat[:, :, :, 2] = (dat[:, :, :, 2] - 114.38170191) / 56.81986099
    return dat


def loadmdl():
    tf.keras.backend.clear_session()
    tf.keras.Model.run_eagerly = True
    model = Net()
    # 加载网络模型参数权重
    model.load_weights(os.path.join(path_cur, "weights.h5"))
    return model


def predict(model, data):
    pred = pd.DataFrame()
    pred["id"] = [x.split(".")[0] for x in ilist]
    # 推理并构造CSV
    pred["label"] = (model(data, False).numpy().ravel() + 0.5).astype(int)
    classes = {0: "neg", 1: "pos"}
    pred["label"] = pred["label"].map(classes)
    return pred


path_cur = os.path.dirname(sys.argv[0])
path_img = sys.argv[1]
path_sub = sys.argv[2]
ilist = os.listdir(path_img)

if __name__ == "__main__":
    data = loaddat()
    model = loadmdl()
    predict(model, data).to_csv(path_sub, index=False)
