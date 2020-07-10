import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,Activation,BatchNormalization,LeakyReLU,Input,Conv3D,Add


def convbn(x,filters=128,kernel_size=3,strides=1,padding="same",dilation_rate=1,batchnorm=False):
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,dilation_rate=dilation_rate)(x)
    # x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
    #            kernel_initializer=kernel_initializer)(x)
    if batchnorm:
        x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.2)(x)
    return x

def conv3bn(x,filters=128,kernel_size=3,strides=1,padding="same",batchnorm=True):
    x=Conv3D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)(x)
    if batchnorm:
        x=BatchNormalization()(x)
    x=LeakyReLU()(x)
    return x

def resblock(x,filters=64,kernel_size=3,strides=1,padding="same",kernel_initializer="he_normal"):
    x_shortcut=x
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)(x)
    x=LeakyReLU(alpha=0.01)(x)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)(x)
    #x must be the same dimension of x_shortcut
    x=Add()([x,x_shortcut])
    return x
