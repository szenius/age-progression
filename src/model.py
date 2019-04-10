from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, LeakyReLU, Dropout
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

    siamese_output = Dense(1, activation='sigmoid')(distance)

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
    encoding_output = Conv2D(8, 8, activation='linear')(encoding_input)
    encoding_output = Dropout(0.5)(encoding_output)
    encoding_output = LeakyReLU()(encoding_output)
    encoding_output = MaxPool2D(strides=2)(encoding_output)
    encoding_output = Flatten()(encoding_output)
    encoding_output = Dense(8, activation='linear', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(encoding_output)
    encoding_output = Dropout(0.5)(encoding_output)
    encoding_output = LeakyReLU()(encoding_output)
    return Model(inputs=encoding_input, outputs=encoding_output)

# def cnn1(shape):
#     input = Input(shape=shape)
#     output = Conv2D()(input)
