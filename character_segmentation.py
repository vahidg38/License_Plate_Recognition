import os
from skimage import io
from skimage.filters import threshold_otsu
import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import plate_detection

license_plate = np.invert(plate_detection.plate_like_objects[0])
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

        # resize the characters to 100X60 and then append each character into the characters list
        resized_char = resize(roi, (100, 60))
        characters.append(resized_char)

        # this is just to keep track of the arrangement of the characters
        column_list.append(x0)

plt.show()
