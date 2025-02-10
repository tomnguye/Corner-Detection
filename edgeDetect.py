import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage.transform import hough_line, hough_line_peaks, rescale, probabilistic_hough_line
from skimage import feature
from skimage.draw import line 
from skimage.color import rgb2gray
from skimage import io
from skimage.io import imread 

original = imread("Images/DoorTest2.JPG")
scale = 400 / original.shape[0]
image = rescale(original, (scale, scale, 1))
print(image.shape)
image = rgb2gray(image)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image)
edges2 = feature.canny(image, sigma=2)

tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
h, theta, d = hough_line(edges1, theta=tested_angles)
lines = probabilistic_hough_line(edges1, threshold=10, line_length=200, line_gap=60)

# display results
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('noisy image', fontsize=16)

ax[1].imshow(edges1, cmap='gray')
ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=16)

ax[2].imshow(edges2, cmap='gray')
ax[2].set_title(r'Canny filter, $\sigma=2$', fontsize=16)

ax[3].imshow(edges2, cmap='gray')
ax[3].set_title(r'Canny filter Hough', fontsize=16)

for a in ax:
    a.axis('off')

# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
#     ax[3].axline((x0, y0), slope=np.tan(angle + np.pi / 2))
for line in lines:
    p0 , p1 = line 
    ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]))
fig.tight_layout()
plt.show()