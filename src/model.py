from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Dense, Reshape
from keras.layers import LeakyReLU, BatchNormalization
from plot_helper import image_shape
    
def get_cnn_model(shape=(image_shape()[0],image_shape()[1],1)):
    a = Input(shape=shape)
    b = Conv2D(8, 3)(a)
    b = Conv2D(16, 3)(b)
    b = MaxPool2D(strides=2)(b)
    b = Flatten()(b)
    b = Dense(shape[0]*shape[1])(b)
    b = Reshape(shape)(b)
    return Model(inputs=a, outputs=b)

def get_mlp_model():
    pass

