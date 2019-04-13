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
from keras.utils import plot_model
import csv

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
    # Save metrics
    save_metrics(history, time_taken, num_epoch, batch_size, "metrics_epoch{}_batch{}.txt".format(epochs_trained, batch_size))
    # Save model
    save_model(model, 'model_epoch{}_batch{}.h5'.format(epochs_trained, batch_size))

def train_mlp(x, y, num_epoch, batch_size):
    ### Cross Validation ###
    # Get and compile model
    model = mlp()
    model.summary()
    optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    # Shuffle data
    x, y = unison_shuffled_copies(x, y, dual_input=False)
    x_split, y_split = split_data(x, y, chunks=True)
    # Train model
    start_time = time()
    history = []
    for i in range(5): 
        x_train, y_train = collect_train_data(x_split, y_split, i)
        x_test, y_test = x_split[i], y_split[i]
        history.append(model.fit(x=np.array(x_train), y=np.array(y_train), epochs=num_epoch, verbose=1, validation_data=(np.array(x_test), np.array(y_test)), shuffle=True, batch_size=batch_size).history)
        model = mlp()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    end_time = time()
    time_taken = end_time - start_time
    print("Took {} to train over {} epochs and with batch size {}".format(end_time - start_time, num_epoch, batch_size))
    # Compute mean metrics
    loss_mean, val_loss_mean, loss_stdev, val_loss_stdev = compute_desc_stats(history, 'loss', 'val_loss')
    acc_mean, val_acc_mean, acc_stdev, val_acc_stdev = compute_desc_stats(history, 'acc', 'val_acc')
    recall_mean, val_recall_mean, recall_stdev, val_recall_stdev = compute_desc_stats(history, 'recall', 'val_recall')
    precision_mean, val_precision_mean, precision_stdev, val_precision_stdev = compute_desc_stats(history, 'precision', 'val_precision')
    f1_mean, val_f1_mean, f1_stdev, val_f1_stdev = compute_desc_stats(history, 'f1_score', 'val_f1_score')
    # Plot graphs
    plot_graph(loss_mean, val_loss_mean, 'loss', 'Loss', "loss_epoch{}_batch{}_mlp.png".format(num_epoch, batch_size), 100)
    plot_graph(acc_mean, val_acc_mean, 'accuracy', 'Accuracy', "acc_epoch{}_batch{}_mlp.png".format(num_epoch, batch_size), 101)
    plot_graph(recall_mean, val_recall_mean, 'recall', 'Recall', "rec_epoch{}_batch{}_mlp.png".format(num_epoch, batch_size), 102)
    plot_graph(precision_mean, val_precision_mean, 'precision', 'Precision', "pre_epoch{}_batch{}_mlp.png".format(num_epoch, batch_size), 103)
    plot_graph(f1_mean, val_f1_mean, 'f1', 'F1 Score', "f1_epoch{}_batch{}_mlp.png".format(num_epoch, batch_size), 104)  
    # Save metrics
    save_metrics_ext(acc_mean[-1], recall_mean[-1], precision_mean[-1], f1_mean[-1], val_acc_mean[-1], val_recall_mean[-1], val_precision_mean[-1], val_f1_mean[-1], acc_stdev[-1], recall_stdev[-1], precision_stdev[-1], f1_stdev[-1], val_acc_stdev[-1], val_recall_stdev[-1], val_precision_stdev[-1], val_f1_stdev[-1], time_taken, num_epoch, batch_size, "metrics_epoch{}_batch{}_mlp.txt".format(num_epoch, batch_size))
    ### Learning Curve ###
    model = mlp()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    lc_history = []
    i = 1
    size = int(len(x)/100*i)
    while i < 101 and size < len(x):
        lc_history.append(model.fit(x=np.array(x[:size]), y=np.array(y[:size]), epochs=num_epoch, verbose=1, shuffle=True, batch_size=batch_size).history)
        model = mlp()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
        i = i + 1
        size = int(len(x)/100*i)
    save_lc(lc_history, "lc_epoch{}_batch{}_mlp.csv".format(num_epoch, batch_size))
    # Save model
    save_model(model, 'model_epoch{}_batch{}_mlp.h5'.format(num_epoch, batch_size))

def save_lc(history, filename):
    acc = collect_last_metric(history, 'acc')
    f1 = collect_last_metric(history, 'f1_score')
    with open('lc.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(acc)
        writer.writerow(f1)

def collect_last_metric(maps, key):
    out = []
    for map in maps:
        out.append(map[key][-1])
    return out

def compute_desc_stats(history, train_key, test_key):
    train_data, test_data = [], []
    for i in range(len(history)):
        train_data.append(history[i][train_key])
        test_data.append(history[i][test_key])
    return np.mean(np.array(train_data), axis=0), np.mean(np.array(test_data), axis=0), np.std(np.array(train_data), axis=0), np.std(np.array(test_data), axis=0)

def collect_train_data(x_split, y_split, exclude_index):
    x_out = []
    y_out = []
    for i in range(len(x_split)):
        if i != exclude_index:
            x_out.extend(x_split[i])
            y_out.extend(y_split[i])
    return x_out, y_out

def save_metrics_ext(acc_mean, recall_mean, precision_mean, f1_mean, val_acc_mean, val_recall_mean, val_precision_mean, val_f1_mean, acc_stdev, recall_stdev, precision_stdev, f1_stdev, val_acc_stdev, val_recall_stdev, val_precision_stdev, val_f1_stdev, time_taken, num_epoch, batch_size, filename):
    with open(filename, 'w') as f:
        f.write("took {} seconds to train over {} batch size and {} epochs".format(time_taken, batch_size, num_epoch))
        f.write("\ninsample mean: acc {}, recall {}, precision {}, f1 {}".format(acc_mean, recall_mean, precision_mean, f1_mean))
        f.write("\noutsample mean: acc {}, recall {}, precision {}, f1 {}".format(val_acc_mean, val_recall_mean, val_precision_mean, val_f1_mean))
        f.write("\ninsample stdev: acc {}, recall {}, precision {}, f1 {}".format(acc_stdev, recall_stdev, precision_stdev, f1_stdev))
        f.write("\noutsample stdev: acc {}, recall {}, precision {}, f1 {}".format(val_acc_stdev, val_recall_stdev, val_precision_stdev, val_f1_stdev))

def save_metrics(history, time_taken, num_epoch, batch_size, filename):
    with open(filename, 'w') as f:
        f.write("took {} seconds to train over {} batch size and {} epochs".format(time_taken, batch_size, num_epoch))
        f.write("\ninsample: acc {}, recall {}, precision {}, f1 {}".format(history.history['acc'][-1], history.history['recall'][-1], history.history['precision'][-1], history.history['f1_score'][-1]))
        f.write("\noutsample: acc {}, recall {}, precision {}, f1 {}".format(history.history['val_acc'][-1], history.history['val_recall'][-1], history.history['val_precision'][-1], history.history['val_f1_score'][-1]))

def split_data(x, y, split=0.3, dual_input=False, chunks=False, k=5):
    num_test = int(len(y) * split)
    num_train = len(y) - num_test
    if dual_input:
        return [x[0][:num_train],x[1][:num_train]], [x[0][num_train:],x[1][num_train:]], y[:num_train], y[num_train:]
    elif chunks:
        chunk_size = int(len(x)/k)
        i = 0
        x_out = []
        y_out = []
        for j in range(k-1):
            x_out.append(x[i:i+chunk_size])
            y_out.append(y[i:i+chunk_size])
            i = i + chunk_size
        x_out.append(x[i:])
        y_out.append(y[i:])
        return x_out, y_out
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
