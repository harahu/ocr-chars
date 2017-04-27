import chars_deep
import sliding_window
import matplotlib.image as im
import numpy as np
import candidate_select
import ocr_display

img_path = 'detection-images/detection-2.jpg'

import pickle

img = im.imread(img_path) / 255
patches = sliding_window.sliding_window_patches(img)
x = chars_deep.classify_chars(patches.reshape(-1, 20*20))
#pickle.dump(x, open("patches.pickle", 'wb'))
#x = pickle.load(open("patches.pickle", 'rb'))
x = np.dstack(x)[0]
x = x.reshape((*patches.shape[:2], 2))
x = candidate_select.posision_annotate(x)
x = candidate_select.filter_under_treshold(x)
x = candidate_select.non_max_suppress(x)


boxes = (((x,y), (20,20), chr(ord('A')+int(cls))) for ((y,x), (cls, _)) in x)

ocr_display.display(img_path, boxes)
