# imports
import tensorflow as tf

import pickle as pic
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization

# constants
inp_img_dim = [100, 100, 4]
target_img_dim = 10000 # 100 * 100 flattened
global_weights_path = "./weights/inter_model/variables/variables"

# 4 residual blocks
inp = Input(shape=(100, 100, 16))
x = BatchNormalization()(inp)
x = Activation("relu")(x)
x = Conv2D(16, 3,padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(16, 3,padding="same")(x)
out = add([x, inp])
res_block_16 = Model(inputs=inp, outputs=out)

inp = Input(shape=(100, 100, 16))
x = BatchNormalization()(inp)
x = Activation("relu")(x)
x = Conv2D(16, 3,padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(16, 3,padding="same")(x)
out = add([x, inp])
res_block_16_1 = Model(inputs=inp, outputs=out)

inp = Input(shape=(100, 100, 16))
x = BatchNormalization()(inp)
x = Activation("relu")(x)
x = Conv2D(16, 3,padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(32, 3,padding="same")(x)
if not inp.shape[-1] == 32:
  #project with 1x1 convolution
  con = Conv2D(32,1)(inp)
out = add([x, con])
res_block_32 = Model(inputs=inp, outputs=out)

#residual module
inp = Input(shape=(100, 100, 32))
x = BatchNormalization()(inp)
x = Activation("relu")(x)
x = Conv2D(16, 3,padding="same")(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Conv2D(64, 3,padding="same")(x)
if not inp.shape[-1] == 64:
  #project with 1x1 convolution
  con = Conv2D(64,1)(inp)
out = add([x, con])
res_block_64 = Model(inputs=inp, outputs=out)

def getGlobalModel():
    '''
        create global model instance and load it with pre-trained weights
    '''
    # create global model
    inp = Input(shape=(inp_img_dim))
    x_a = Conv2D(16, 3, padding='same')(inp)
    x_a = res_block_16(x_a)
    x_a = res_block_16_1(x_a)
    x_a = res_block_32(x_a)
    x_a = res_block_64(x_a)

    x_a = MaxPooling2D(7)(x_a) # reduce parameters, only feed forward max activation in 7*7 window

    x_a = Flatten()(x_a) # flatten and feed to dense layer
    x_a = Dense(1024, activation='relu')(x_a) # intermediate layer tp reduce parameters
    out = Dense(target_img_dim, activation='softmax')(x_a)
    # create model
    model = Model(inputs=inp, outputs=out)
    # load pre-trained model weights
    model.load_weights(global_weights_path)

    return model # return fully trained model
