from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Dense, Reshape, BatchNormalization, ReLU
from helper import image_shape
    
def get_cnn_model(shape=(image_shape()[0],image_shape()[1],1)):
    a = Input(shape=shape)
    b = Conv2D(64, 3)(a)
    b = BatchNormalization()(a)
    b = Conv2D(128, 3)(b)
    b = BatchNormalization()(a)
    b = MaxPool2D(strides=2)(b)
    b = Flatten()(b)
    b = Dense(shape[0]*shape[1])(b)
    b = Reshape(shape)(b)
    return Model(inputs=a, outputs=b)

def get_feature_model(shape=(image_shape()[0],image_shape()[1],1)):
    img = Input(shape=shape, name='feature_model_input')
    t = Conv2D(16, 3, activation='linear')(img)
    t = BatchNormalization()(t)
    t = ReLU()(t)
    t = Conv2D(32, 3, activation='linear')(t)
    t = BatchNormalization()(t)
    t = ReLU()(t)
    t = MaxPool2D(strides=2)(t)
    t = Flatten()(t)
    t = Dense(32, activation='linear')(t)
    t = BatchNormalization()(t)
    t = ReLU()(t)
    return Model(inputs=img, outputs=t, name='feature_model')

def get_mlp_model():
    pass

