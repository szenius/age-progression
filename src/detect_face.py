'''
Adapted from: https://stackoverflow.com/a/42408508
'''

from PIL import Image
from skimage import io
from os import listdir
from os.path import isfile, join
from plot_helper import image_shape
import dlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import random


def get_images(dir_path, load_saved=False):
    if load_saved is True:
        before, after = load_batch("{}/{}/".format(dir_path, "before")), load_batch("{}/{}/".format(dir_path, "after"))
    else:
        before, after = extract_faces(dir_path)
    before = add_augmented_images(before)
    after = add_augmented_images(after)
    return np.array(before), np.array(after)

def add_augmented_images(images):
    results = []
    for i, image in enumerate(images):
        results.append(image)
        results.append(flip_img(image))
        results.append(rotate_img(image, 90))
        results.append(rotate_img(image, 180))
        results.append(rotate_img(image, 270))
        # results.append(translate_left_img(image))
        # results.append(translate_right_img(image))
        # results.append(translate_up_img(image))
        # results.append(translate_down_img(image))
        results.append(add_noise_img(image))
        print("Applied augmentation to image {}/{}".format(i, len(images)))
    return results

def flip_img(image):
    return np.fliplr(image)

def rotate_img(image, degree):
    num_rotations = int(degree / 90)
    for i in range(num_rotations):
        image = np.rot90(image)
    return image

def translate_left_img(image):
    for i in range(image.shape[1], 1, -1):
        for j in range(image.shape[0]):
            if (i < image.shape[1]-20):
                image[j][i] = image[j][i-20]
            elif (i < image.shape[1]-1):
                image[j][i] = 0
    return image

def translate_right_img(image):
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if (i < image.shape[1]-20):
                image[j][i] = image[j][i+20]
    return image

def translate_up_img(image):
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if (j < image.shape[0] - 20 and j > 20):
                image[j][i] = image[j+20][i]
            else:
                image[j][i] = 0
    return image

def translate_down_img(image):
    for j in range(image.shape[0], 1, -1):
        for i in range(image.shape[1]):
            if (j < 144 and j > 20):
                image[j][i] = image[j-20][i]
    return image

def add_noise_img(image):
    noise = np.random.randint(5, size = image.shape, dtype = 'uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if (image[i][j][k] != 255):
                    image[i][j][k] += noise[i][j][k]
    return image

def load_batch(dir_path):
    images = []
    for f in listdir(dir_path):
        file_path = join(dir_path, f)
        if isfile(file_path) and file_path.endswith(".png"):
            image = Image.open(file_path)
            image = process_image(image)
            images.append(image)
            print("Loaded {} from {}".format(file_path, dir_path))
    return images

def extract_faces(dir_path):
    images = read_images(dir_path)
    before, after = [], []
    for i, image in enumerate(images):
        detected_faces = detect_faces(image)
        if len(detected_faces) != 2: 
            # Don't consider cases where we cannot extract exactly two faces
            continue 
        detected_faces.sort(key=lambda tup: tup[0])
        before_img, after_img = generate_image(detected_faces[0], image), generate_image(detected_faces[1], image)
        save_images(dir_path, str(i), before_img[:,:,0], after_img[:,:,0])
        before.append(before_img)
        after.append(after_img)
    return before, after

def generate_negative_egs(pos_x, pos_y, neg_eg_ratio=1):
    neg_x, neg_y = [], []
    for i in range(len(pos_x)):
        neg_indices = []
        while len(neg_indices) < neg_eg_ratio:
            neg_idx = generate_diff_index(len(pos_x), i)
            while neg_idx in neg_indices:
                neg_idx = generate_diff_index(len(pos_x), i)
            neg_indices.append(neg_idx)
            neg_x.append(pos_x[i])
            neg_y.append(pos_y[neg_idx])
    return np.array(neg_x), np.array(neg_y)

def generate_diff_index(n, i):
    rand = random.randint(0, n-1)
    while (rand == i):
        rand = random.randint(0, n-1)
    return rand

def generate_image(face, original_image):
    image = Image.fromarray(original_image).crop(face) # Crop image
    return process_image(image)

def process_image(image, shape=image_shape()):
    image = image.resize(shape, Image.ANTIALIAS) # Resize image
    image = image.convert('L') # Convert to grayscale
    image = np.array(image) # Convert to numpy array
    image = image / 255.0 # Normalize image
    return np.expand_dims(image, 3)

def detect_faces(image):
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]
    return face_frames

def read_images(dir_path):
    images = []
    for f in listdir(dir_path):
        file_path = join(dir_path, f)
        if isfile(file_path) and file_path.endswith(".png") or file_path.endswith(".jpg"):
            images.append(io.imread(file_path))
            print("Read image {} from {}".format(file_path, dir_path))
    return images

def save_images(dir_path, image_name, before_img, after_img):
    plt.imsave("{}/{}/{}.png".format(dir_path, "before", image_name), before_img)
    plt.imsave("{}/{}/{}.png".format(dir_path, "after", image_name), after_img)
