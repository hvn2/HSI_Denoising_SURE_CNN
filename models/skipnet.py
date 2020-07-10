import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, BatchNormalization,Conv3D
from tensorflow.keras.layers import LeakyReLU,Concatenate, Add
from models.common import *
""""Define SkipNet structure used in the paper
    Iput:   ndown: number of convolution blocks (K in the paper)
            channel: output channel
    Ouput:  model
    """
def skip(ndown=3,channel=191):
    #down side
    input_layer = Input((None,None,channel))
    out = input_layer
    skips=[]
    for i in range(2):
        out = convbn(out,dilation_rate=2)
        skips.append(convbn(out,filters=64,kernel_size=1))
    for i in range(2, ndown):
        out = convbn(out, dilation_rate=1)
        skips.append(convbn(out, filters=64, kernel_size=1))
    skips.reverse()
    for i in range(ndown):
        if i==0:
            out=convbn(out)
        else:
            out = convbn(Concatenate()([out,skips[i]]))
    # out = convbn(out)

    out = Conv2D(input_layer.shape[-1],1,activation="sigmoid")(out)
    # out = Conv2D(input_layer.shape[-1],1)(out)

    mymodel=Model(input_layer,out)
    return mymodel
"""Define a skip network for subspace denoising (do not use conv at output layer)"""
def skipsubspace(ndown=3,channel=191):
    #down side
    input_layer = Input((None,None,channel))
    out = input_layer
    skips=[]
    for i in range(2):
        out = convbn(out,dilation_rate=2)
        skips.append(convbn(out,filters=64,kernel_size=1))
    for i in range(2, ndown):
        out = convbn(out, dilation_rate=1)
        skips.append(convbn(out, filters=64, kernel_size=1))
    skips.reverse()
    for i in range(ndown):
        if i==0:
            out=convbn(out)
        else:
            out = convbn(Concatenate()([out,skips[i]]))

    out = Conv2D(input_layer.shape[-1],1)(out)

    mymodel=Model(input_layer,out)
    return mymodel

"""Define a skip network with downsampling and upsampling (modified UNet)"""
def skipdown(ndown=5,channel=191):
    def down_layer(X):
        '''
        inpupt: X (batch, h,w,c)
        downsampling by conv stride =2,128 filters, kernelsize=3
        return: X
        '''
        X = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(X)
        X = Conv2D(128, 3, strides=2)(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)

        X = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(X)
        X = Conv2D(128, 3)(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)
        return X

    def up_layer(X):
        '''
        input: X (batch,h,w,c)
        2 conv layer 128 filter, kernel size =3, and upsampling layer bilinear
        return: X
        '''
        X = BatchNormalization()(X)
        X = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(X)
        X = Conv2D(128, 3)(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)

        X = Conv2D(128, 1)(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)
        X = UpSampling2D(interpolation='bilinear')(X)
        return X

    def skip_layer(X):
        '''
        Skip connenction layer: 1 conv layer with 4 filter, kernel size 1
        input: X
        return: X
        '''
        X = Conv2D(4, 1)(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.2)(X)
        return X

    input_layer = Input((None,None,channel))
    out = input_layer
    skips = []
    for i in range(ndown):
        out = down_layer(out)
        skips.append(skip_layer(out))
    skips.reverse()
    # up side
    for i in range(ndown):
        if i==0:
            out=up_layer(skips[0])
        else:
            out = up_layer(Concatenate()([out, skips[i]]))

    out = Conv2D(channel, 1, activation="sigmoid")(out)
    model = Model(input_layer, out)
    return model

"""Define Residual network
Input:  K: number of residual block
        channelin: number of input channel
        channelout: number of output channel"""
def residualnet(K=24,channelin=191,channelout=191):
    input_layer = Input((None,None,channelin))
    out = input_layer
    out=Conv2D(64,3,padding="same")(out)
    out=LeakyReLU(alpha=0.01)(out)
    first_out=out
    for i in range(K):
        out=resblock(out)

    out=Conv2D(64,3,padding="same")(out)
    out=LeakyReLU(alpha=0.01)(out)
    out=Conv2D(64,3,padding="same")(out)
    out=Add()([first_out,out])
    out=Conv2D(128,3,padding="same")(out)
    out=LeakyReLU(alpha=0.01)(out)
    out=Conv2D(256,3,padding="same")(out)
    out=LeakyReLU(alpha=0.01)(out)
    out=Conv2D(filters=channelout,kernel_size=3,padding="same")(out)
    out_layer=Add()([input_layer,out])
    out_layer=Conv2D(channelout,1,activation="sigmoid")(out_layer)

    model=Model(input_layer,out_layer)
    return model