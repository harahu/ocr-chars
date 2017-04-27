from itertools import product
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np

def map_over_windows(func, img):
    patches = extract_patches_2d(img, (20,20))
    patches = patches.reshape(img.shape[0] - 20, img.shape[1] - 20, 20 * 20)
    return np.apply_along_axis(func, 2, patches)
