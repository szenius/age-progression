from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Dense, Reshape, BatchNormalization, ReLU, Lambda, K
from plot_helper import image_shape
    
def siamese_net(shape=(image_shape()[0],image_shape()[1],1)):
    left_input = Input(shape=shape)
    right_input = Input(shape=shape)

    # Image Encoding Model
    encoding_input = Input(shape=shape)
    encoding_output = Conv2D(16, 3, activation='linear')(encoding_input)
    encoding_output = BatchNormalization()(encoding_output)
    encoding_output = ReLU()(encoding_output)
    encoding_output = Conv2D(32, 3, activation='linear')(encoding_output)
    encoding_output = BatchNormalization()(encoding_output)
    encoding_output = ReLU()(encoding_output)
    encoding_output = MaxPool2D(strides=2)(encoding_output)
    encoding_output = Flatten()(encoding_output)
    encoding_output = Dense(8, activation='linear')(encoding_output)
    encoding_output = BatchNormalization()(encoding_output)
    encoding_output = ReLU()(encoding_output)
    encoding_model = Model(inputs=encoding_input, outputs=encoding_output)

    # Encode input images
    left_encoding = encoding_model(left_input)
    right_encoding= encoding_model(right_input)

    # Comput absolute difference between encodings
    DifferenceLayer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    distance = DifferenceLayer([left_encoding, right_encoding])

    # Add layer to compute similarity score
    siamese_output = Dense(1, activation='sigmoid')(distance)

    return Model(inputs=[left_input, right_input], outputs=siamese_output)