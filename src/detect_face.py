'''
Adapted from: https://stackoverflow.com/a/42408508
'''

from PIL import Image
from skimage import io
from os import listdir
from os.path import isfile, join
from helper import plot_images, image_shape
import dlib
import matplotlib.pyplot as plt
import numpy as np
import sys


def get_images(dir_path, load_saved=False):
    if load_saved is True:
        return load_batch("{}/{}/".format(dir_path, "before")), load_batch("{}/{}/".format(dir_path, "after"))
    else:
        return extract_faces(dir_path)

def load_batch(dir_path):
    images = []
    for f in listdir(dir_path):
        file_path = join(dir_path, f)
        if isfile(file_path) and file_path.endswith(".png"):
            image = Image.open(file_path)
            image = process_image(image)
            images.append(image)
    return np.array(images)

def extract_faces(dir_path):
    images = read_images(dir_path)
    before, after = [], []
    for i, image in enumerate(images):
        detected_faces = detect_faces(image)
        if len(detected_faces) != 2: 
            # Don't consider cases where we cannot extract exactly two photos
            continue 
        detected_faces.sort(key=lambda tup: tup[0])
        before_img, after_img = generate_image(detected_faces[0], image), generate_image(detected_faces[1], image)
        save_images(dir_path, str(i), before_img[:,:,0], after_img[:,:,0])
        before.append(before_img)
        after.append(after_img)
    return np.array(before), np.array(after)

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
        if isfile(file_path) and file_path.endswith(".png"):
            images.append(io.imread(file_path))
    return images

def save_images(dir_path, image_name, before_img, after_img):
    plt.imsave("{}/{}/{}.png".format(dir_path, "before", image_name), before_img)
    plt.imsave("{}/{}/{}.png".format(dir_path, "after", image_name), after_img)
