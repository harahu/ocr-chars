import itertools
from sliding_window import stride

def strided_range(end):
    return range(0, end*stride, stride)

def cartesian(*ranges):
    return itertools.product(*map(strided_range, ranges))

def posision_annotate(img):
    image_size = (img.shape[0] - 19, img.shape[1] - 19)
    img = img.reshape(-1, 2)
    return zip(cartesian(*image_size), img)

def filter_under_treshold(annotated_imgs):
    return filter(lambda x: x[1][1] > 0.50, annotated_imgs)

def non_max_suppress(annotated_imgs):
    imgs = sorted(annotated_imgs)

    i = 0
    while i < len(imgs):
        j = i+1
        tx,ty = imgs[i][0]
        while j < len(imgs):
            # check_overlap
            x,y = imgs[j][0]
            if -20 < x-tx < 20 and -20 < y-ty < 20:
                del imgs[j]
            j += 1
        i += 1

    return imgs
