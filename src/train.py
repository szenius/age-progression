from detect_face import get_images, generate_negative_egs
from model import *
from plot_helper import *
from keras import optimizers, losses, Model, callbacks
from tensorflow import set_random_seed
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np
import sys
from keras import backend as K
from time import time
import keras_metrics

plt.set_cmap('gray')

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

def unison_shuffled_copies(a, b):
    img1 = a[0]
    img2 = a[1]
    assert img1.shape[0] == img2.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[1])
    return [img1[p], img2[p]], b[p]

def train(x1, x2, y, num_epoch, batch_size, model_type='diff'):
    # Get and compile model
    model = siamese_net()
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    # Shuffle and split data
    x, y = unison_shuffled_copies(np.array([x1, x2]), y)
    train_x, train_y, test_x, test_y = split_data(x, y)
    # Train model
    start_time = time()
    history = model.fit(x=x, y=y, epochs=num_epoch, verbose=1, validation_split=0.3, shuffle=True, batch_size=batch_size, callbacks=[callbacks.EarlyStopping(monitor='val loss', patience=2)])
    end_time = time()
    epochs_trained = len(history.history['loss'])
    print("Took {} to train over {} epochs with batch size {}".format(end_time - start_time, epochs_trained, batch_size))
    # Plot graphs
    plot_graph(history.history['loss'], history.history['val_loss'], 'loss', 'Loss', "loss_epoch{}_batch{}.png".format(epochs_trained, batch_size))
    plot_graph(history.history['acc'], history.history['val_acc'], 'accuracy', 'Accuracy', "acc_epoch{}_batch{}.png".format(epochs_trained, batch_size))
    plot_graph(history.history['recall'], history.history['val_recall'], 'recall', 'Recall', "rec_epoch{}_batch{}.png".format(epochs_trained, batch_size))
    plot_graph(history.history['precision'], history.history['val_precision'], 'precision', 'Precision', "pre_epoch{}_batch{}.png".format(epochs_trained, batch_size))
    plot_graph(history.history['f1_score'], history.history['val_f1_score'], 'f1', 'F1 Score', "f1_epoch{}_batch{}.png".format(epochs_trained, batch_size))  
    # Save model
    save_model(model, 'model_epoch{}_batch{}.h5'.format(epochs_trained, batch_size))

def split_data(x, y, split=0.8, filename='indices.txt'):
    split_exists = os.path.isfile(filename)
    if split_exists:
        with open(filename) as f:
            indices = [[int(x) for x in line.strip().split()] for line in f]
    else:
        k = len(y) * split
        indices = random.sample(xrange(len(data)), k)
        with open(filename, 'w') as f:
            for item in indices:
                f.write("%d " % item)
    x0 = [x[0][i] for i in indices]
    x1 = [x[1][i] for i in indices]
    y = [y[i] for i in indices]
    return [x0, x1], y

def save_model(model, filename):
    model.save(filename)

def run(dir_path, load_saved, num_epoch, neg_eg_ratio, batch_size, model_type):
    pos_x1, pos_x2, neg_x1, neg_x2 = load_data(dir_path, load_saved, neg_eg_ratio)
    pos_y, neg_y = np.ones(pos_x1.shape[0]), np.zeros(neg_x1.shape[0])
    x1 = np.vstack([pos_x1, neg_x1])
    x2 = np.vstack([pos_x2, neg_x2])
    y = np.hstack([pos_y, neg_y])
    train(x1, x2, y, num_epoch, batch_size, model_type=model_type)

def get_boolean_val(input):
    return input.lower() == "true"

if __name__ == '__main__':
    run(sys.argv[1], get_boolean_val(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])