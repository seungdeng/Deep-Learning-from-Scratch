#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


img = imread('cactus.jpg')

plt.imshow(img)
plt.show()