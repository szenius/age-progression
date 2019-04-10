from matplotlib import pyplot as plt

    
def image_shape():
    return (150,150)

def plot_loss(loss, val_loss, filename):
    plot_graph(loss, val_loss, "loss", "Loss", filename)

def plot_accuracy(acc, val_acc, filename):
    plot_graph(acc, val_acc, "acc", "Accuracy", filename)

def plot_specificity(spec, val_spec, filename):
    plot_graph(spec, val_spec, "spec", "Specificity", filename)

def plot_sensitivity(sen, val_sen, filename):
    plot_graph(sen, val_sen, "sen", "Sensitivity", filename)

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