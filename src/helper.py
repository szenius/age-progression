from matplotlib import pyplot as plt

    
def image_shape():
    return (180,180)

def plot_images(original, predicted, actual):
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(original)
    axarr[1].imshow(predicted)
    axarr[2].imshow(actual)
    plt.savefig("result.png")

def save_image(path, arr):
    plt.imsave(path, arr)