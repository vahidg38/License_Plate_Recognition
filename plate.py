import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import os
from skimage import io
from skimage.filters import threshold_otsu


l=io.imread('/home/vahid/Desktop/lpr/plates/p1.jpg', as_gray=True)
l= l * 255
threshold_value = threshold_otsu(l)
b = l > threshold_value
license_plate = np.invert(b)

license_plate = resize(license_plate,(110,600))
labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1,1)
ax1.imshow(license_plate, cmap="gray")

character_dimensions = (0.5*license_plate.shape[0], 1*license_plate.shape[0], 0.05*license_plate.shape[1], 1*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

characters = []
counter=0
column_list = []
for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        # draw a red bordered rectangle over the character.
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                       linewidth=1, fill=False)
        ax1.add_patch(rect_border)

        # resize the characters to 20X20 and then append each character into the characters list
        resized_char = resize(roi, (100, 60))
        characters.append(resized_char)

        # this is just to keep track of the arrangement of the characters
        column_list.append(x0)

current_dir = os.path.dirname(os.path.realpath(__file__))
save_directory = os.path.join(current_dir, 'test')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

cv2.imwrite('test/zzw.jpg',characters[0]*1000)
cv2.imwrite('test/zzw2.jpg',characters[1]*1000)
cv2.imwrite('test/zzw3.jpg',characters[2]*1000)
cv2.imwrite('test/zzwyzzzz4.jpg',characters[3]*1000)
cv2.imwrite('test/zzwyzzzz5.jpg',characters[4]*1000)
cv2.imwrite('test/zzwyzzzz6.jpg',characters[5]*1000)
#cv2.imwrite('test/z26g6.jpg',characters[6]*1000)


plt.show()
