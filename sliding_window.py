from sklearn.feature_extraction.image import extract_patches_2d

def sliding_window_patches(img):
    """Create an array of contents of a 20x20 window sliding over img with
    stride 1 in both directions."""
    patches = extract_patches_2d(img, (20,20))
    patches = patches.reshape((img.shape[0] - 20) * (img.shape[1] - 20), 20 * 20)
    return patches
