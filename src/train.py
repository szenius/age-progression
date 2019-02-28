from detect_face import get_images
from model import get_cnn_model, get_mlp_model
from helper import *
from keras import optimizers, losses
from tensorflow import set_random_seed
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np
import sys

plt.set_cmap('gray')

def train(dir_path, load_saved):
    # Prepare dataset
    x, y = get_images(dir_path, load_saved)
    print("Loaded {} images from \"{}\".".format(x.shape[0], dir_path))
    x_eval, y_eval = np.array([x[-1]]), np.array([y[-1]])
    x, y = x[:-1], y[:-1]

    # Train model
    results = []
    model = get_cnn_model()
    optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    history = model.fit(x=x, y=y, epochs=1000, verbose=1, validation_data=(x_eval, y_eval))
    results.append(history)

    # Plot loss
    fig = plt.figure(300)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    filename = "loss.png"
    plt.savefig(filename)

    # Test filter on eval data
    pred_y = model.predict(x_eval)
    plot_images(x_eval[0,:,:,0], pred_y[0,:,:,0], y_eval[0,:,:,0])

if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2])