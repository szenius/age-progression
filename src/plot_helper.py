from matplotlib import pyplot as plt

    
def image_shape():
    return (150,150)

def plot_graph(train_metric, val_metric, key_short, key_long, filename, id):
    plt.figure(id)
    plt.plot(train_metric, label='train {}'.format(key_short))
    plt.plot(val_metric, label='val {}'.format(key_short))
    plt.xlabel("epoch")
    plt.ylabel(key_short)
    plt.title("Train vs Validation {}".format(key_long))
    plt.legend()
    plt.savefig(filename)

def plot_roc(fpr, tpr, filename, id=1):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig(filename)        
    
def save_image(path, arr):
    plt.imsave(path, arr)