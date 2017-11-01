from __future__ import division
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte, util, data, io, filters, exposure, img_as_float, img_as_ubyte, feature
from skimage.morphology import disk
import skimage.morphology as mp
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
from numpy import array
from IPython.display import display
from ipywidgets import *
from ipykernel.pylab.backend_inline import flush_figures


def load_image(file_location):
    filename = os.path.join(os.path.dirname(__file__), file_location)
    image = data.load(filename, as_grey=True)
    return image


def load_all_images_from_directory():
    image = [];
    for i in range(1):
        image.append(load_image("images/kosci" + str(i) + ".jpg"))
    return image


def display_images(images):
    for i in images:
        plt.imshow(i, cmap='gray')

    plt.xticks([])
    plt.yticks([])
    plt.show()


images = load_all_images_from_directory()
display_images(images)
