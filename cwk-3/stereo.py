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
canny_vals = [90, 158, 64, 5]

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
    global disparityNormalised, disparity_img
    canny_images = [cv2.Canny(i, canny_vals[0], canny_vals[1]) for i in images]
    disparity_img = disparity.getDisparityMap(canny_images[0], canny_images[1], canny_vals[2], canny_vals[3])
    disparityNormalised = np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityNormalised)
    picture = np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 255.0))
    cv2.imwrite(f'./img/output/Disparity.png', picture)
    cv2.imwrite(f'./img/output/CannyL.png', canny_images[0])
    cv2.imwrite(f'./img/output/CannyR.png', canny_images[1])


def change1(x):
    canny_vals[0] = x
    getDisparity()


def change2(x):
    canny_vals[1] = x
    getDisparity()


def change1a(x):
    canny_vals[2] = x
    getDisparity()


def change2a(x):
    canny_vals[3] = x
    getDisparity()


def task2():
    original_photo = images[0]
    copy_photo = images[0]
    depth_image = np.zeros_like(disparityNormalised)
    for a, r in enumerate(disparityNormalised):
        for b, c in enumerate(r):
            pass


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
disparity_img = []
cv2.createTrackbar('Canny1', 'Disparity', canny_vals[0], 255, change1)
cv2.createTrackbar('Canny2', 'Disparity', canny_vals[1], 255, change2)
cv2.createTrackbar('Number of disparities', 'Disparity', canny_vals[2], 255, change1a)
cv2.createTrackbar('Block Size', 'Disparity', canny_vals[3], 255, change2a)

pause()

# VIEWS OF THE SCENE
image_depth = np.zeros_like(disparityNormalised)

# calculate Z value for each coordinate
coordinates = []
for i, row in enumerate(disparity_img):
    for j, column in enumerate(row):

        # depth = baseline*(focal_length/(disparity + doffs))
        image_depth[i][j] = baseline * (cam0[0][0] / (disparity_img[i][j] + doffs))

        if disparityNormalised[i][j] > 0:
            # Calculate world co-ordinates using similar triangles
            # x = (depth/focus) * X0
            x = (column / focal_length) * i
            coordinates.append((x, j, image_depth[i][j]))

# Plot disparity
disparity.plot(coordinates)
