from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Dense, Reshape, BatchNormalization, ReLU, concatenate
from helper import image_shape
    
def siamese_net(shape=(image_shape()[0],image_shape()[1],1)):
    img_x1 = Input(shape=shape, name='input x1')
    img_x2 = Input(shape=shape, name 'input x2')
    img_x1_features = feature_gen_cnn(img_x1)
    img_x2_features = feature_gen_cnn(img_x2)
    combined = concatenate([img_x1_features, img_x2_features], name='combined')
    combined = Dense(16, activation='linear')(combined)
    combined = BatchNormalization()(combined)
    combined = ReLU(combined)
    combined = Dense(4, activation='linear')(combined)
    combined = BatchNormalization()(combined)
    combined = ReLU()(combined)
    combined = Dense(1, activation='sigmoid')(combined)
    return Model(inputs=[img_x1, img_x2], outputs=combined, name='siamese_net')

def feature_gen_cnn(shape=(image_shape()[0],image_shape()[1],1)):
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

