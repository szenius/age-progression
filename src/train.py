from detect_face import get_images, generate_negative_egs
from model import *
from plot_helper import *
from keras import optimizers, losses, Model, callbacks
from tensorflow import set_random_seed
from matplotlib import pyplot as plt
from PIL import Image
from keras import backend as K
from time import time
import keras_metrics
import os
import random
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np
import sys
from sklearn.metrics import roc_curve

plt.set_cmap('gray')
np.random.seed(0)

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def load_data(dir_path, load_saved, neg_eg_ratio):
    before, after = get_images(dir_path, load_saved)
    print("Loaded {} pairs of images from \"{}\".".format(before.shape[0], dir_path))
    neg_before, neg_after = generate_negative_egs(before, after, neg_eg_ratio=neg_eg_ratio)
    return before, after, neg_before, neg_after

def unison_shuffled_copies(a, b, dual_input=True):
    if dual_input:
        img1 = a[0]
        img2 = a[1]
        assert img1.shape[0] == img2.shape[0] == b.shape[0]
        p = np.random.permutation(a.shape[1])
        return [img1[p], img2[p]], b[p]
    else:
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

def train_cnn(x1, x2, y, num_epoch, batch_size):
    # Get and compile model
    model = siamese_net()
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    # Shuffle data
    x, y = unison_shuffled_copies(np.array([x1, x2]), y)
    x_train, x_test, y_train, y_test = split_data(x, y, dual_input=True)
    # Train model
    start_time = time()
    history = model.fit(x=x_train, y=y_train, epochs=num_epoch, verbose=1, validation_data=(x_test, y_test), shuffle=True, batch_size=batch_size, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=2)])
    end_time = time()
    time_taken = end_time - start_time
    epochs_trained = len(history.history['loss'])
    print("Took {} to train over {} epochs with batch size {}".format(end_time - start_time, epochs_trained, batch_size))
    # Plot train vs test metric graphs
    plot_graph(history.history['loss'], history.history['val_loss'], 'loss', 'Loss', "loss_epoch{}_batch{}.png".format(epochs_trained, batch_size), 200)
    plot_graph(history.history['acc'], history.history['val_acc'], 'accuracy', 'Accuracy', "acc_epoch{}_batch{}.png".format(epochs_trained, batch_size), 201)
    plot_graph(history.history['recall'], history.history['val_recall'], 'recall', 'Recall', "rec_epoch{}_batch{}.png".format(epochs_trained, batch_size), 202)
    plot_graph(history.history['precision'], history.history['val_precision'], 'precision', 'Precision', "pre_epoch{}_batch{}.png".format(epochs_trained, batch_size), 203)
    plot_graph(history.history['f1_score'], history.history['val_f1_score'], 'f1', 'F1 Score', "f1_epoch{}_batch{}.png".format(epochs_trained, batch_size), 204)  
    # Plot ROC curve
    y_pred = model.predict(x_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plot_roc(fpr, tpr, "roc_epoch{}_batch{}.png".format(epochs_trained, batch_size))
    # Save metrics
    save_metrics(history, fpr, tpr, time_taken, epoch_trained, batch_size, "metrics_epoch{}_batch{}.txt".format(epochs_trained, batch_size))
    # Save model
    save_model(model, 'model_epoch{}_batch{}.h5'.format(epochs_trained, batch_size))

def train_mlp(x, y, num_epoch, batch_size):
    # Get and compile model
    model = mlp()
    model.summary()
    optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    # Shuffle data
    x, y = unison_shuffled_copies(x, y, dual_input=False)
    x_train, x_test, y_train, y_test = split_data(x, y)
    # Train model
    start_time = time()
    history = model.fit(x=x_train, y=y_train, epochs=num_epoch, verbose=1, validation_data=(x_test, y_test), shuffle=True, batch_size=batch_size, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)])
    end_time = time()
    epoch_trained = len(history.history['loss'])
    print("Took {} to train over {} epochs and with batch size {}".format(end_time - start_time, epoch_trained, batch_size))
    # Plot graphs
    plot_graph(history.history['loss'], history.history['val_loss'], 'loss', 'Loss', "loss_epoch{}_batch{}_mlp.png".format(epoch_trained, batch_size), 100)
    plot_graph(history.history['acc'], history.history['val_acc'], 'accuracy', 'Accuracy', "acc_epoch{}_batch{}_mlp.png".format(epoch_trained, batch_size), 101)
    plot_graph(history.history['recall'], history.history['val_recall'], 'recall', 'Recall', "rec_epoch{}_batch{}_mlp.png".format(epoch_trained, batch_size), 102)
    plot_graph(history.history['precision'], history.history['val_precision'], 'precision', 'Precision', "pre_epoch{}_batch{}_mlp.png".format(epoch_trained, batch_size), 103)
    plot_graph(history.history['f1_score'], history.history['val_f1_score'], 'f1', 'F1 Score', "f1_epoch{}_batch{}_mlp.png".format(epoch_trained, batch_size), 104)  
    # Plot ROC curve
    y_pred = model.predict(x_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plot_roc(fpr, tpr, "roc_epoch{}_batch{}_mlp.png".format(epoch_trained, batch_size))
    # Save metrics
    save_metrics(history, fpr, tpr, "metrics_epoch{}_batch{}_mlp.txt".format(epoch_trained, batch_size))
    # Save model
    save_model(model, 'model_epoch{}_batch{}_mlp.h5'.format(epoch_trained, batch_size))

def save_metrics(history, fpr, tpr, time_taken, epoch_trained, batch_size, filename):
    with open(filename, 'w') as f:
        f.write("took {} seconds to train over {} batch size and {} epochs".format(time_taken, batch_size, epoch_trained))
        f.write("insample: acc {}, recall {}, precision {}, f1 {}".format(history.history['acc'][-1], history.history['recall'][-1], history.history['precision'][-1], history.history['f1_score'][-1]))
        f.write("\noutsample: acc {}, recall {}, precision {}, f1 {}".format(history.history['val_acc'][-1], history.history['val_recall'][-1], history.history['val_precision'][-1], history.history['val_f1_score'][-1]))
        f.write("\nfpr: ")
        for item in fpr:
            f.write("{} ".format(item))
        f.write("\ntpr: ")
        for item in tpr:
            f.write("{} ".format(item))

def split_data(x, y, split=0.3, dual_input=False):
    num_test = int(len(y) * split)
    num_train = len(y) - num_test
    if dual_input:
        return [x[0][:num_train],x[1][:num_train]], [x[0][num_train:],x[1][num_train:]], y[:num_train], y[num_train:]
    else:
        return x[:num_train], x[num_train:], y[:num_train], y[num_train:]

def save_model(model, filename):
    model.save(filename)

def run_cnn(dir_path, load_saved, num_epoch, neg_eg_ratio, batch_size):
    pos_x1, pos_x2, neg_x1, neg_x2 = load_data(dir_path, load_saved, neg_eg_ratio)
    pos_y, neg_y = np.ones(pos_x1.shape[0]), np.zeros(neg_x1.shape[0])
    x1 = np.vstack([pos_x1, neg_x1])
    x2 = np.vstack([pos_x2, neg_x2])
    y = np.hstack([pos_y, neg_y])
    train_cnn(x1, x2, y, num_epoch, batch_size)

def run_mlp(dir_path, num_epoch, neg_eg_ratio, batch_size):
    x, y = load_feature_data(dir_path)
    train_mlp(x, y, num_epoch, batch_size)

def load_feature_data(dir_path):
    feature_data = np.genfromtxt(dir_path, delimiter=',', skip_header=1)
    labels = feature_data[:,-2]
    feature_data = feature_data[:,0:-2]
    return feature_data, labels

def get_boolean_val(input):
    return input.lower() == "true"

if __name__ == '__main__':
    mode = sys.argv[6]
    if mode == 'cnn':
        run_cnn(sys.argv[1], get_boolean_val(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif mode == 'mlp':
        run_mlp(sys.argv[1], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        print("You asked for mode {} but there is only 'cnn' and 'mlp'".format(mode))
