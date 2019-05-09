import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

#https://en.wikipedia.org/wiki/Kernel_(image_processing)
kernel = [
    [[0,-1,0],[-1,5,-1],[0,-1,0]],
    [[0,0,0],[0,1,0],[0,0,0]],
    [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
][0]

#Open the image, and turn it into a grayscale (1 channel) 8-bit image
img = Image.open('grayscale.jpg').convert('L')
plt.imshow(img, cmap="gray")
plt.show()

convolved = convolve2d(img, kernel)
plt.imshow(convolved, cmap="gray")
plt.show()
