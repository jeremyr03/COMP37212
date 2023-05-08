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

img = [f"./img/input/girl{i}.png" for i in ('L', 'R')]


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


def sobel():
    gradients = [
        (cv2.convertScaleAbs(
            cv2.Sobel(cv2.GaussianBlur(i, (3, 3), cv2.BORDER_DEFAULT),
                      cv2.CV_16S,
                      1, 0, ksize=3, scale=1, delta=0,
                      borderType=cv2.BORDER_DEFAULT)),
         cv2.convertScaleAbs(
             cv2.Sobel(cv2.GaussianBlur(i, (3, 3), cv2.BORDER_DEFAULT),
                       cv2.CV_16S,
                       0, 1, ksize=3, scale=1, delta=0,
                       borderType=cv2.BORDER_DEFAULT)))
        for i in images
    ]

    return [cv2.addWeighted(i[0], 0.5, i[1], 0.5, 0) for i in gradients]


def getDisparity():
    global disparityNormalised, disparity_img
    # canny_images = [cv2.Canny(i, canny_vals[0], canny_vals[1]) for i in images]
    canny_images = images
    cv2.imshow('Sobel', canny_images[0])
    disparity_img = disparity.getDisparityMap(canny_images[0], canny_images[1], canny_vals[2], canny_vals[3])
    disparityNormalised = np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 1.0))
    # cv2.imshow('Disparity', disparityNormalised)
    picture = np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 255.0))


def change1(x):
    canny_vals[0] = x
    getDisparity()
    cv2.imshow('Disparity', disparityNormalised)
    cv2.imwrite(f'./img/output/Disparity.png',
                np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 255.0)))


def change1a(x):
    canny_vals[2] = x
    getDisparity()
    cv2.imshow('Disparity', disparityNormalised)
    cv2.imwrite(f'./img/output/Disparity.png',
                np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 255.0)))


def change2a(x):
    canny_vals[3] = x
    getDisparity()
    cv2.imshow('Disparity', disparityNormalised)
    cv2.imwrite(f'./img/output/Disparity.png',
                np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 255.0)))


def change2(x):
    canny_vals[1] = x
    getDisparity()
    cv2.imshow('Disparity', disparityNormalised)
    cv2.imwrite(f'./img/output/Disparity.png',
                np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 255.0)))


def change3(x):
    canny_vals[0] = x
    getDisparity()
    task2()


def change4(x):
    canny_vals[1] = x
    getDisparity()
    task2()


def depth(x):
    global d
    d = x
    task2()


def thresh(x):
    global t
    t = x
    task2()


def task2():
    global t
    original_photo = cv2.imread(img[0])
    blurred_photo = cv2.medianBlur(original_photo, 21)
    grey_photo = cv2.cvtColor(images[0], cv2.COLOR_GRAY2RGB)
    depth_image = np.zeros_like(disparityNormalised)
    for a, r in enumerate(disparityNormalised):
        for b, c in enumerate(r):
            # depth = 1/(disparity + k)
            depth_image[a][b] = 1 / (disparity_img[a][b] + d)

    normalised_depth = np.interp(depth_image, (depth_image.min(), depth_image.max()), (0.0, 255.0))
    for a, r in enumerate(original_photo):
        for b, c in enumerate(r):
            if normalised_depth[a][b] > t:
                original_photo[a][b] = blurred_photo[a][b]
    cv2.imshow(f"Depth", original_photo)


images = [open_image(img[0]), open_image(img[1])]
# cv2.imshow(f"blur", images[0])

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
cv2.createTrackbar('Canny1', 'Disparity', canny_vals[0], 1000, change1)
cv2.createTrackbar('Canny2', 'Disparity', canny_vals[1], 1000, change2)
cv2.createTrackbar('Number of disparities', 'Disparity', canny_vals[2], 255, change1a)
cv2.createTrackbar('Block Size', 'Disparity', canny_vals[3], 255, change2a)

pause()

# VIEWS OF THE SCENE
# depth = baseline*(focal_length/(disparity + doffs))
image_depth = np.zeros_like(disparityNormalised)

# calculate Z value for each coordinate
coordinates = []
for i, row in enumerate(disparity_img):
    for j, column in enumerate(row):
        image_depth[i][j] = baseline * (focal_length / (disparityNormalised[i][j] + doffs))
        if disparityNormalised[i][j] > 0:
            # Calculate world co-ordinates using similar triangles
            # x = (depth/focus) * X0
            x = (column / focal_length) * i
            coordinates.append((x, j, image_depth[i][j]))

# Plot disparity
disparity.plot(coordinates)

# TASK 2

# Calculate Depth
cv2.imshow(f"Depth", disparityNormalised)
img_depth = disparityNormalised
d = 0
t = 128
# cv2.createTrackbar('Canny1', 'Depth', canny_vals[0], 1000, change3)
cv2.createTrackbar('Canny2', 'Depth', canny_vals[1], 1000, change4)
cv2.createTrackbar('Depth', 'Depth', 0, 255, depth)
cv2.createTrackbar('Threshold', 'Depth', 120, 255, thresh)
pause()
