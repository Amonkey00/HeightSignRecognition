import tensorflow as tf
import numpy as np
import os
import cv2 as cv


def processImg(img_path):
    img = cv.imread(img_path)
    img = cv.resize(img,(112,112))
    img = img / 255
    return img


def getTrainDataset(img_dir):
    x_train = []
    y_train = []
    corrects = os.listdir(os.path.join(img_dir,'Correct'))
    wrongs = os.listdir(os.path.join(img_dir,'Wrong'))
    for imgDir in corrects:
        x_train.append(processImg(os.path.join(img_dir,'Correct',imgDir)))
        y_train.append([1])
    wrong_idx = np.random.randint(0,len(wrongs),3*len(corrects))
    for idx in wrong_idx:
        x_train.append((processImg(os.path.join(img_dir,'Wrong',wrongs[idx]))))
        y_train.append([0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = tf.convert_to_tensor(x_train,tf.float32)
    y_train = tf.convert_to_tensor(y_train,tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    return dataset


def getTestDataset(img_dir):
    x_train = []
    y_train = []
    wrongs = os.listdir(os.path.join(img_dir,'Wrong'))

    wrong_idx = np.random.randint(0,len(wrongs),100)
    for idx in wrong_idx:
        x_train.append((processImg(os.path.join(img_dir,'Wrong',wrongs[idx]))).tolist())
        y_train.append([0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train,y_train
    # x_train = tf.convert_to_tensor(x_train,tf.float32)
    # y_train = tf.convert_to_tensor(y_train,tf.float32)
    # dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    # return dataset


