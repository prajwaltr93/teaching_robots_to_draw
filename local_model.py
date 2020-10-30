from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# force graph building by disabling eager execution
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

weights_path = "./weights/local_model_weights/local_model_trained_1"

# build , intiliaze with trained weights and return model for inference
def getLocalModel():
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

    #residual module
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

    """Building Model

    4 Block Residual block with dynamic tensor extraction

    two input, two output model (keras functional API)
    """

    # tensor slice
    def slice_return(x):
      return tf.slice(x[0], x[1], [5,5,64])

    #creating local model
    inp = Input(shape = inp_img_dim, dtype=tf.float32, name = "lg_inp")
    ext_inp = Input(shape = inp_ext_dim, dtype = tf.int32, name = "ext_inp")

    # initilization layer, helpful in transfer learning, due to different input layers, between global and local models
    conv = Conv2D(16, 3, padding='same')(inp)
    #four residual block stacked
    x_a = res_block_16(conv)
    x_a = res_block_16_1(x_a)
    x_a = res_block_32(x_a)
    x_a = res_block_64(x_a)

    # now x is 95 * 65 * 54 res encoded tensor, now carry out extraction procedure to enforce localization
    # batch_size is Dynamic, Unknown or of type None, hence using map_fn to iterate over dimention '0' --> None
    extracted_tensor = tf.map_fn(slice_return, elems = (x_a, ext_inp), fn_output_signature=tf.float32)
    x_0 = Flatten()(extracted_tensor) #flatten and feed to dense layer
    #fully connected layer 1
    # x1 = Dense(256, activation='relu')(x_0)
    x1 = Dense(128, activation='relu')(x_0)
    #fully connected layer 2
    x2 = Dense(1, activation='sigmoid', name = 'out_touch')(x1)
    #fully connected layer 3
    x3 = Dense(25, activation='softmax', name = 'out_cropped')(x1)
    # x3 = tf.reshape(x3, (5,5)) #output a 5 * 5 image
    model = Model(inputs= [inp, ext_inp], outputs= [x2,x3])

    # initialize with trained weights
    model.load_weights(weights_path)

    return model
