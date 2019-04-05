from matplotlib import pyplot as plt

    
def image_shape():
    return (150,150)

def plot_images(original, predicted, actual, file_name="result"):
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(original)
    axarr[1].imshow(predicted)
    axarr[2].imshow(actual)
    plt.savefig("{}.png".format(file_name))

def plot_loss(loss, val_loss, filename):
    fig = plt.figure(300)
    plt.plot(loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig(filename)

def save_image(path, arr):
    plt.imsave(path, arr)