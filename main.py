from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from pylab import *
import skimage as ski
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte, util, data, io, filters, exposure, img_as_float, img_as_ubyte, feature, data, io, filters, exposure
from skimage.morphology import disk
import skimage.morphology as mp
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
from numpy import array
from IPython.display import display
from ipywidgets import *
from ipykernel.pylab.backend_inline import flush_figures

NUMBER_OF_IMAGES = 1

#load one image from file
def load_image(file_location):
    filename = os.path.join(os.path.dirname(__file__), file_location)
    image = data.load(filename, as_grey=False)
    return image

#collecting many images from file to table
def load_all_images_from_directory():
    image = [];
    for i in range(NUMBER_OF_IMAGES):
        image.append(load_image("images/kosci" + str(i) + ".jpg"))
    return image

#display all images from list
def display_images(images):
    for i in images:
        plt.imshow(i, cmap='gray')

    plt.xticks([])
    plt.yticks([])
    plt.show()

#collecting images to convert many images
def convert_all_images(images):
    converted_images = []
    for i in images:
        converted_images.append(convert_image(i))
    return converted_images

#function to convert single image
def convert_image(image):
    image = rgb2gray(image)
    image = feature.canny(image, 2.5)
    return image

#function to plot stats of single image
def plot_hist(image):
    histo, x = np.histogram(image, range(0, 256), density=True)
    plt.plot(histo)
    xlim(0, 255)
    plt.show()


images = load_all_images_from_directory()
display_images(images)
plot_hist(images[0])
images = convert_all_images(images)
