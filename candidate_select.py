import itertools

def cartesian(*ranges):
    return itertools.product(*map(range, ranges))

def posision_annotate(img):
    image_size = (img.shape[0] - 20, img.shape[1] - 20)
    return zip(cartesian(*image_size), img)

def filter_under_treshold(annotated_imgs):
    return filter(lambda x: x[1][1] > 0.7, annotated_imgs)

def non_max_suppress(annotated_imgs):
    imgs = sorted(annotated_imgs)

    i = 0
    while i < len(imgs):
        j = i
        tx,ty = imgs[i][0]
        while j < len(imgs):
            # check_overlap
            x,y = imgs[j][0]
            if -20 < x-tx < 20 and -20 < y-ty < 20:
                del imgs[j]

    return imgs
