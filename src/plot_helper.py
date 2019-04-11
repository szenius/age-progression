from matplotlib import pyplot as plt

    
def image_shape():
    return (150,150)

def plot_loss(loss, val_loss, filename):
    plot_graph(loss, val_loss, "loss", "Loss", filename)

def plot_accuracy(acc, val_acc, filename):
    plot_graph(acc, val_acc, "accuracy", "Accuracy", filename)

def plot_recall(recall, val_recall, filename):
    plot_graph(recall, val_recall, "recall", "Recall", filename)

def plot_precision(prec, val_prec, filename):
    plot_graph(prec, val_prec, "precision", "Precision", filename)

def plot_graph(train_metric, val_metric, key_short, key_long, filename):
    plt.plot(train_metric, label='train {}'.format(key_short))
    plt.plot(val_metric, label='val {}'.format(key_short))
    plt.xlabel("epoch")
    plt.ylabel(key_short)
    plt.title("Train vs Validation {}".format(key_long))
    plt.legend()
    plt.savefig(filename)
    
def save_image(path, arr):
    plt.imsave(path, arr)