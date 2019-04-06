from detect_face import get_images, generate_negative_egs
from model import *
from plot_helper import *
from keras import optimizers, losses, Model
from tensorflow import set_random_seed
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np
import sys

plt.set_cmap('gray')

def load_data(dir_path, load_saved, neg_eg_ratio):
    x, y = get_images(dir_path, load_saved)
    print("Loaded {} images from \"{}\".".format(x.shape[0], dir_path))
    num_eval = int(x.shape[0] * 0.3)
    x_eval, y_eval = x[-num_eval:], y[-num_eval:]
    x, y = x[:-num_eval], y[:-num_eval]
    neg_x, neg_y = generate_negative_egs(x, y, neg_eg_ratio=neg_eg_ratio)
    return x, y, neg_x, neg_y, x_eval, y_eval

def predict(model, x_eval, y_eval):
    pred = model.predict(x_eval)
    for i in range(len(x_eval)):
        plot_images(x_eval[i][:,:,0], pred[i][:,:,0], y_eval[i][:,:,0], file_name="pred_" + str(i))

def train(pos_x, pos_y, neg_x, neg_y, x_eval, y_eval, num_epoch, model='cnn'):
    if model == 'cnn':
        x = np.vstack([pos_x, neg_x])
        y = np.vstack([pos_y, neg_y])
        model = get_cnn_model()
        optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        history = model.fit(x=x, y=y, epochs=num_epoch, verbose=1, validation_data=(x_eval, y_eval), shuffle=True)
        plot_loss(history.history['loss'], history.history['val_loss'], "loss.png")
        predict(model, x_eval, y_eval)
        save_model(model, 'model.h5')
    else:
        pass # todo: stub

def save_model(model, filename):
    model.save(filename)

def run(dir_path, load_saved, num_epoch, neg_eg_ratio):
    pos_x, pos_y, neg_x, neg_y, x_eval, y_eval = load_data(dir_path, load_saved, neg_eg_ratio)
    train(pos_x, pos_y, neg_x, neg_y, x_eval, y_eval, num_epoch)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))