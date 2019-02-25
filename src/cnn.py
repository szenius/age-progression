from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras import optimizers, losses
from tensorflow import set_random_seed
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from PIL import Image
import numpy as np

plt.set_cmap('gray')

def save_image(path, arr):
    plt.imsave(path, arr)
    
def load_image(path, shape=(224,224)):
    im = Image.open(path)
    im = im.resize(shape, Image.ANTIALIAS)
    im = im.convert('L')
    return np.array(im) / 255.0

def get_model(shape=(224,224,1)):
    a = Input(shape=shape)
    b = Conv2D(32, 5)(a)
    b = Conv2DTranspose(1, 5, output_shape=shape)(b)
    return Model(inputs=a, outputs=b)

def plot_images(original, predicted, actual):
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(original)
    axarr[1].imshow(predicted)
    axarr[2].imshow(actual)
    plt.show()

# Load dataset
testset_path = "./data/testset/"
original = []
progressed = []
for f in listdir(testset_path):
    file_path = join(testset_path, f)
    if isfile(file_path) and file_path.endswith("_1.png"):
        image, image_f = load_image(file_path), load_image(file_path.replace("_1.png", "_2.png"))
        image = np.expand_dims(image, 3)
        image_f = np.expand_dims(image_f, 3)
        original.append(image)
        progressed.append(image_f)
x = np.array(original)
y = np.array(progressed)
x_eval, y_eval = np.array([x[-1]]), np.array([y[-1]])
x, y = x[:-2], y[:-2]

# Train model
results = []
model = get_model()
optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=optimizer)
history = model.fit(x=x, y=y, epochs=1000, verbose=1, validation_data=(x_eval, y_eval))
results.append(history)

# Plot loss
fig = plt.figure(300)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Train vs Validation Loss")
plt.legend()
filename = "loss.png"
plt.savefig(filename)

# Test filter on eval data
pred_y = model.predict(x_eval)
plot_images(x_eval[0,:,:,0], pred_y[0,:,:,0], y_eval[0,:,:,0])