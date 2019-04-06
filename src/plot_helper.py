from matplotlib import pyplot as plt

    
def image_shape():
    return (150,150)

def plot_loss(loss, val_loss, filename):
    fig = plt.figure(300)
    plt.plot(loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig(filename)

def plot_accuracy(acc, val_acc, filename):
    fig = plt.figure(301)
    plt.plot(acc, label='train acc')
    plt.plot(val_acc, label='val acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.title("Train vs Validation Accurary")
    plt.legend()
    plt.savefig(filename)

def save_image(path, arr):
    plt.imsave(path, arr)