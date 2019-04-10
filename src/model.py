from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, LeakyReLU, Dropout, ReLU
from keras.layers import Dense, Reshape, BatchNormalization, ReLU, Lambda, K, concatenate, regularizers
from plot_helper import image_shape
    
def siamese_net(shape=(image_shape()[0],image_shape()[1],1)):
    left_input = Input(shape=shape)
    right_input = Input(shape=shape)

    # Image Encoding Model
    encoding_model = get_encoding_model(shape)

    # Encode input images
    left_encoding = encoding_model(left_input)
    right_encoding= encoding_model(right_input)

    # Compute absolute difference between encodings
    DifferenceLayer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    distance = DifferenceLayer([left_encoding, right_encoding])

    # 1st Dense Layer
    siamese_output = Dense(128)(distance)
    siamese_output = ReLU()(siamese_output)
    siamese_output = Dropout(0.4)(siamese_output)
    siamese_output = BatchNormalization()(siamese_output)

    # 2nd Dense Layer
    siamese_output = Dense(128)(siamese_output)
    siamese_output = ReLU()(siamese_output)
    siamese_output = Dropout(0.4)(siamese_output)
    siamese_output = BatchNormalization()(siamese_output)

    # 3rd Dense Layer
    siamese_output = Dense(20)(siamese_output)
    siamese_output = ReLU()(siamese_output)
    siamese_output = Dropout(0.4)(siamese_output)
    siamese_output = BatchNormalization()(siamese_output)

    siamese_output = Dense(1, activation='sigmoid')(siamese_output)

    return Model(inputs=[left_input, right_input], outputs=siamese_output)

def siamese_net_concat(shape=(image_shape()[0],image_shape()[1],1)):
    left_input = Input(shape=shape)
    right_input = Input(shape=shape)

    # Image Encoding Model
    encoding_model = get_encoding_model(shape)

    # Encode input images
    left_encoding = encoding_model(left_input)
    right_encoding = encoding_model(right_input)

    # Concatenate encodings
    combined = concatenate([left_encoding, right_encoding])

    # Add layer to compute similarity score
    siamese_output = Dense(1, activation='sigmoid')(combined)

    return Model(inputs=[left_input, right_input], outputs=siamese_output)

def get_encoding_model(shape):
    encoding_input = Input(shape=shape)

    # 1st Convolutional Layer
    encoding_output = Conv2D(filters=96, kernel_size=11, strides=4, padding='valid')(encoding_input)
    encoding_output = ReLU()(encoding_output)
    encoding_output = MaxPool2D(pool_size=2, strides=2, padding='valid')(encoding_output)
    encoding_output = BatchNormalization()(encoding_output)

    # 2nd Convolutional Layer
    encoding_output = Conv2D(filters=256, kernel_size=11, strides=1, padding='valid')(encoding_input)
    encoding_output = ReLU()(encoding_output)
    encoding_output = MaxPool2D(pool_size=2, strides=2, padding='valid')(encoding_output)
    encoding_output = BatchNormalization()(encoding_output)

    # 3rd Convolutional Layer
    encoding_output = Conv2D(filters=384, kernel_size=3, strides=1, padding='valid')(encoding_input)
    encoding_output = ReLU()(encoding_output)
    encoding_output = BatchNormalization()(encoding_output)

    # 4th Convolutional Layer
    encoding_output = Conv2D(filters=384, kernel_size=3, strides=1, padding='valid')(encoding_input)
    encoding_output = ReLU()(encoding_output)
    encoding_output = BatchNormalization()(encoding_output)

    # 5th Convolutional Layer
    encoding_output = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid')(encoding_input)
    encoding_output = ReLU()(encoding_output)
    encoding_output = MaxPool2D(pool_size=2, strides=2, padding='valid')(encoding_output)
    encoding_output = BatchNormalization()(encoding_output)

    # Flatten
    encoding_output = Flatten()(encoding_output)

    return Model(inputs=encoding_input, outputs=encoding_output)