import tensorflow as tf
print(tf.__version__)

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
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import Sequence

#data generator
def data_gen(file_list,batch_size):
    #training loop is infinite, stops after steps_per_epoch is complete
    while True:
        for file in file_list:
            #load data
            data = pic.load(open(file,'rb'),encoding="bytes")
            cur_len = len(data['sG_data'])
            for i in range(0,cur_len,batch_size):
                x = data['sG_data'][i:i+batch_size]
                y = data['sG_labels'][i:i+batch_size]
                yield x,y
                x = [] #clear contents
                y = []

#residual module
def res_module(inp):
    if not inp.shape[-1] == 64:
        #project with 1x1 convolution
        inp = Conv2D(64,1,activation="relu")(inp)
    x = BatchNormalization()(inp)
    x = Activation("relu")(x)
    x = Conv2D(16, 3,padding="same",activation="sigmoid")(x)
    x = Conv2D(32, 3,padding="same",activation="sigmoid")(x)
    x = Conv2D(64, 3,padding="same",activation="sigmoid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(16, 3,padding="same",activation="sigmoid")(x)
    x = Conv2D(32, 3,padding="same",activation="sigmoid")(x)
    x = Conv2D(64, 3,padding="same",activation="sigmoid")(x)
    out = add([x, inp])
    return out

def getGlobalModel():
    #flour block residual module
    inp = Input(shape=(95, 60, 4))
    x_a = res_module(inp)
    x_a = res_module(x_a)
    x_a = res_module(x_a)
    x_a = res_module(x_a)
    x_a = MaxPooling2D(7)(x_a)
    x_a = Flatten()(x_a)
    out = Dense(5700,activation='softmax')(x_a)
    # create model
    model = Model(inputs=inp, outputs=out)

    return model

def checkModel(model):
    #data files folder
    data_files_folder_path = "/home/starkm42/hershey_datset/global_dataset/"
    no_files = 2
    files_train = [data_files_folder_path + "data_batch_"+i.__str__() for i in range(no_files)] #two files

    x,y = next(data_gen(files_train,100))
    pred_img = model.predict(x)
    actual_imgs = y

    for pred,img in zip(pred_img,actual_imgs):
      print("predicted : ",np.argmax(pred),"actual : ",np.argmax(img),"pred : ",pred[np.argmax(pred)]," actual : ",img[np.argmax(img)])
