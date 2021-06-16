import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    MaxPool2D,
    Dense,
    add
)
from tensorflow.keras.activations import softmax
from tensorflow.keras import(
    Sequential
)


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self,filter_num,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters=filter_num,
                            kernel_size=(3,3),
                            strides=stride,
                            padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=filter_num,
                            kernel_size=(3,3),
                            strides=1,
                            padding='same')
        self.bn2= BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(Conv2D(filters=filter_num,
                                       kernel_size=(1,1),
                                       strides=stride))
            self.downsample.add(BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self,inputs,training=None,**kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x,training=training)

        output = tf.nn.relu(add([residual, x]))
        return output


def make_resnetBlock(filter_num, blocks,stride=1):
    res_block = Sequential()
    res_block.add(BasicBlock(filter_num,stride=stride))
    for _ in range(1,blocks):
        res_block.add(BasicBlock(filter_num,stride=1))
    return res_block


class PatchNet(Model):
    def __init__(self,num_classes=2):
        super(PatchNet, self).__init__()
        self.conv1 = Conv2D(filters=64,
                            kernel_size=(7,7),
                            strides=2,
                            padding='same')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size=(3,3),
                               strides=2,
                               padding='same')
        self.layer1 = make_resnetBlock(filter_num=64,
                                       blocks=2)
        self.layer2 = make_resnetBlock(filter_num=128,
                                       blocks=2,
                                       stride=2)
        self.layer3 = make_resnetBlock(filter_num=256,
                                       blocks=2,
                                       stride=2)
        self.layer4 = make_resnetBlock(filter_num=512,
                                       blocks=2,
                                       stride=2)
        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(units=num_classes,activation=softmax,kernel_regularizer=tf.keras.regularizers.l2())

    def call(self,inputs,training=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output
