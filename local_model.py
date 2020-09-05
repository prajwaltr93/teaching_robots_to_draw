#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pickle as pic
#data preparation :: training on single dataset
path = '/home/starkm42/hershey_datset/local_dataset/'

fd = open(path + 'data_batch_0', 'rb')
data = pic.load(fd, encoding="bytes")

sample_rate_train = 100
lg_data = data['lG_data'][:sample_rate_train]
lg_extract = data['lG_extract'][:sample_rate_train]
lg_touch = data['lG_touch'][:sample_rate_train]
lg_croppedimg = data['lG_croppedimg'][:sample_rate_train]

# len(lg_data)


# In[2]:


#imports
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
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import Sequence

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


# In[3]:


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


# In[10]:


#creating local model

lG_inp = Input(shape = (95, 65, 3), name = "lg_inp")

ext_inp = Input(shape = (3), dtype = tf.int32, name = "ext_inp")

#defining model

#four residual module stacked 
x = res_module(lG_inp)
x = res_module(x)
x = res_module(x)
x = res_module(x)

# now x is 95 * 65 * 54 res encoded tensor, now carry out extraction procedure to enforce localization
# batch_size is Dynamic, Unknown or of type None, hence using map_fn to iterate over dimention '0' --> None

extracted_tensor = tf.map_fn(lambda x : tf.slice(x[0], x[1], [5,5,64]), elems = (x, ext_inp), dtype = (tf.float32))

x = Flatten()(extracted_tensor) #flatten and feed to dense layer

#fully connected layer 1

x1 = Dense(128, activation='relu')(x)

#fully connected layer 2
x2 = Dense(1, activation='softmax', name = 'out_touch')(x1)

#fully connected layer 3
x3 = Dense(25, activation='softmax', name = 'out_cropped')(x1)

# x3 = tf.reshape(x3, (5,5)) #output a 5 * 5 image

model = Model(inputs= [ lG_inp, ext_inp], outputs= [x2,x3])


# In[11]:


model.summary()


# In[12]:


model.compile(loss=[tf.keras.losses.BinaryCrossentropy(),'categorical_crossentropy'],
              optimizer='adam',
              metrics=['accuracy'])


# In[15]:


model.fit({"lg_inp" : lg_data, "ext_inp" : lg_extract }, { "out_touch" : lg_touch, "out_cropped" : lg_croppedimg}, epochs = 5)


# In[16]:


model.predict({"lg_inp" : lg_data, "ext_inp" : lg_extract })


# In[ ]:




