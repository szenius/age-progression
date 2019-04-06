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
    before, after = get_images(dir_path, load_saved)
    print("Loaded {} pairs of images from \"{}\".".format(before.shape[0], dir_path))
    neg_before, neg_after = generate_negative_egs(before, after, neg_eg_ratio=neg_eg_ratio)
    return before, after, neg_before, neg_after

def train(x1, x2, y, num_epoch):
    model = siamese_net()
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x=[x1, x2], y=y, epochs=num_epoch, verbose=1, validation_split=0.3, shuffle=True)
    plot_loss(history.history['loss'], history.history['val_loss'], "loss.png")
    save_model(model, 'model.h5')

def save_model(model, filename):
    model.save(filename)

def run(dir_path, load_saved, num_epoch, neg_eg_ratio):
    pos_x1, pos_x2, neg_x1, neg_x2 = load_data(dir_path, load_saved, neg_eg_ratio)
    pos_y, neg_y = np.ones(pos_x1.shape[0]), np.zeros(neg_x1.shape[0])
    x1 = np.vstack([pos_x1, neg_x1])
    x2 = np.vstack([pos_x2, neg_x2])
    y = np.hstack([pos_y, neg_y])
    train(x1, x2, y, num_epoch)

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))