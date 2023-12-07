from skimage import io
from skimage.filters import threshold_local
from skimage import segmentation
import imutils
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches



# first we need some preprocess
# first we read the car image in grayscale mode and then mutiply the image by 255 to get pixels in range of(0,255)
car = io.imread('/Users/mac/Desktop/license-plate-recognition/license-plate-recognition/lpr/cars/input6.jpg', as_gray=True)
car_gray = car * 255

fig, (ax1, ax2) = plt.subplots(1, 2)

# plotting car_gray
ax1.imshow(car_gray, cmap="gray")

threshold = threshold_otsu(car_gray)
# using threshold value to creat binary image from grayscale image
car_binary = car_gray > threshold

ax2.imshow(car_binary, cmap="gray")
plt.show()
# we use connected component analysis 
# this gets all the connected regions and groups them together
label_image = measure.label(car_binary)

# getting the maximum width, height and minimum width and height that a license plate can be

plate_dimensions = (0*label_image.shape[0], 0.1*label_image.shape[0], 0.1*label_image.shape[1], 0.3*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []
fig, (ax1) = plt.subplots(1)
ax1.imshow(car, cmap="gray");

# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    if region.area < 30000 or region.area>65000:
        
        continue

    # the bounding box coordinates
    min_row, min_col, max_row, max_col= region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    # ensuring that the region identified satisfies the condition of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        plate_like_objects.append(car_binary[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                              max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=3, fill=False)
        ax1.add_patch(rectBorder)
    

plt.show()
