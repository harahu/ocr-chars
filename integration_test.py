import chars_deep
import sliding_window
import matplotlib.image as im
import numpy as np

img_path = 'detection-images/detection-2.jpg'

img = im.imread(img_path) / 255
patches = sliding_window.sliding_window_patches(img)
print(chars_deep.classify_chars(patches[:100000]))
