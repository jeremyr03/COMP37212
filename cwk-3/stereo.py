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

cam0 = [(5806.559, 0, 1429.219),
        (0, 5806.559, 993.403),
        (0, 0, 1)]
cam1 = [(5806.559, 0, 1543.51),
        (0, 5806.559, 993.403),
        (0, 0, 1)]
doffs = 114.291
baseline = 174.019
width = 2960
height = 2016

img = [f"./img/input/umbrella{i}.png" for i in ('L', 'R')]


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


def canny():
    canny_images = [cv2.Canny(i, canny_vals[0], canny_vals[1]) for i in images]
    disparity_img = disparity.getDisparityMap(canny_images[0], canny_images[1], 64, 5)
    disparityNormalised = np.interp(disparity_img, (disparity_img.min(), disparity_img.max()), (0.0, 1.0))
    cv2.imshow('Disparity', disparityNormalised)
    cv2.imwrite(f'./img/output/Disparity.png', disparityNormalised)


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


def change1(x):
    canny_vals[0] = x
    canny()


def change2(x):
    canny_vals[1] = x
    canny()


images = [open_image(img[0]), open_image(img[1])]
cv2.imshow(f"blur", images[0])

# FOCAL LENGTH CALCULATION


# DISPARITY MAP
# sobel_images = sobel()
cv2.imshow(f"Disparity", images[0])
cv2.createTrackbar('Canny1', 'Disparity', canny_vals[0], 255, change1)
cv2.createTrackbar('Canny2', 'Disparity', canny_vals[1], 255, change2)
change1(canny_vals[0])
change2(canny_vals[1])

while True:
    if cv2.waitKey(1) == ord(' '):
        break

cv2.destroyAllWindows()
