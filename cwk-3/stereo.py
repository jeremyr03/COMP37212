import sys
import cv2
import os
import numpy as np
import scipy
import disparity

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
canny_vals = [90, 158]

underline = "________________________________________________"

cam0 = [(5806.559, 0, 1429.219),
        (0, 5806.559, 993.403),
        (0, 0, 1)]
cam1 = [(5806.559, 0, 1543.51),
        (0, 5806.559, 993.403),
        (0, 0, 1)]
sensor = [22.2, 14.8]
resolution = [3088, 2056]
doffs = 114.291
baseline = 174.019
# width = 2960
# height = 2016
width = 740
height = 505

img = [f"./img/input/umbrella{i}.png" for i in ('L', 'R')]


def pause():
    while True:
        if cv2.waitKey(1) == ord(' '):
            break

    cv2.destroyAllWindows()


def open_image(i=None):
    # Display Image
    if i is None:
        print("Error: Trying to open nothing.")
        sys.exit()
    else:
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f'Error: failed to open {i}')
        sys.exit()

    return image


def getDisparity():
    global disparityNormalised
    canny_images = [cv2.Canny(i, canny_vals[0], canny_vals[1]) for i in images]
    disparity_img = disparity.getDisparityMap(canny_images[0], canny_images[1], 64, 5)
    disparityNormalised = np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityNormalised)
    picture = np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 255.0))
    cv2.imwrite(f'./img/output/Disparity.png', picture)


def change1(x):
    canny_vals[0] = x
    getDisparity()


def change2(x):
    canny_vals[1] = x
    getDisparity()


images = [open_image(img[0]), open_image(img[1])]
cv2.imshow(f"blur", images[0])

# FOCAL LENGTH CALCULATION
# focal length(mm) = focal length (px) / (sensor width (mm) x image width (px))
focal_length = cam0[0][0] * (sensor[0] / resolution[0])
print(f"focal length "
      f"= focal length ({cam0[0][0]}px) * (sensor width ({sensor[0]}mm) / image width ({resolution[0]}mm))"
      f"\nf={focal_length}mm"
      f"\n{underline}")

# DISPARITY MAP
cv2.imshow(f"Disparity", images[0])
disparityNormalised = []
cv2.createTrackbar('Canny1', 'Disparity', canny_vals[0], 255, change1)
cv2.createTrackbar('Canny2', 'Disparity', canny_vals[1], 255, change2)

pause()

# VIEWS OF THE SCENE
# depth = baseline*(focal_length/(disparity + doffs))
image_depth = np.zeros_like(disparityNormalised)

# calculate Z value for each coordinate
coordinates = []
for i, row in enumerate(disparityNormalised):
    for j, column in enumerate(row):
        if disparityNormalised[i][j] > 0.5:
            x = (disparityNormalised[i][j]/focal_length) * i
            coordinates.append((x, j, disparityNormalised[i][j]))
        # image_depth[i][j] = baseline * (focal_length / (disparityNormalised[i][j] + doffs))

# Calculate world co-ordinates using similar triangles
# x = (depth/focus) * X0


# Plot disparity
# print(coordinates)
disparity.plot(coordinates)
