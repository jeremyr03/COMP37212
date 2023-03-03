import sys
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import kernels

SAVE_PATH = "./images/"
SAMPLE_IMAGE = f"{SAVE_PATH}kitty.bmp"
sample_kernel = kernels.MEAN_KERNEL
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
max_value = 255
max_type = 4
max_binary_value = 255
threshold_value = 180


# opening initial image
def open_image(i, filename=None):
    # Display Image
    if not filename:
        filename = "untitled"
    image = cv2.imread(i)

    if image is None:
        print(f'Error: failed to open {i}')
        sys.exit()

    cv2.namedWindow(filename)
    cv2.imshow(filename, image)

    return image


# part 1 - convolution over a kernel
def convolution(image=SAMPLE_IMAGE, kernel=sample_kernel, name="untitled", save=False, show=True):
    new_image = image
    kernel = np.array(kernel)

    # add padding around image
    n = len(kernel)
    padding = int((n - 1) / 2)
    padded_grey = cv2.copyMakeBorder(src=image,
                                     top=padding,
                                     bottom=padding,
                                     left=padding,
                                     right=padding,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=0)

    # loop through each pixel and do a convolution
    for row in range(len(image)):
        for column in range(len(image[0])):
            region = np.array(padded_grey[row:row + n, column:column + n].copy())
            total = np.sum(np.multiply(kernel, region))
            if np.sum(kernel) != 0:
                total = int(total / np.sum(kernel))

            # accounting for overflow & underflow
            if total >= 256:
                total = 255
            elif total < 0:
                total = 0
            new_image[(row - padding)][(column - padding)] = total

    if show:
        cv2.namedWindow(name)
        cv2.imshow(name, new_image)
    if save:
        cv2.imwrite(f"{SAVE_PATH}{name}", new_image)
    return new_image


# part 2 - combine the two convolutions
def combine(image1, image2):
    new_image = image1

    for i in range(len(image1)):
        for j in range(len(image1[0])):
            # find edge magnitude
            magnitude = np.sqrt((image1[i][j] ** 2) + (image2[i][j] ** 2))

            # accounting for overflow (there would be no overflow)
            if magnitude >= 256:
                magnitude = 255

            new_image[i][j] = magnitude

    return new_image


def change(i, experiment, x=0):
    output = threshold(i, x)
    cv2.imshow(f"experiment{experiment}_threshold_{os.path.basename(image_path)}", output)
    histogram(output, f"{SAVE_PATH}experiment{experiment}_histogram.png")
    cv2.imwrite(f"{SAVE_PATH}experiment{experiment}_threshold_{os.path.basename(image_path)}", output)


# part 3 - show threshold of image
def threshold(im, x):
    global threshold_value
    temp = im.copy()
    for i in range(len(temp)):
        for j in range(len(temp[0])):
            if temp[i][j] < x:
                temp[i][j] = 0
            else:
                temp[i][j] = 255

    return temp


def sobel(image, experiment=0, mean="mean"):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{SAVE_PATH}greyscale_{os.path.basename(image_path)}", image)

    options = {"mean": kernels.MEAN_KERNEL,
               "3x3": kernels.GAUSSIAN_KERNEL_3,
               "5x5": kernels.GAUSSIAN_KERNEL_5,
               "7x7": kernels.GAUSSIAN_KERNEL_7
               }
    try:
        k = options[mean]
        # smoothing
        smooth_image = convolution(image=image,
                                   kernel=k,
                                   name=f"experiment{experiment}_{mean}_{os.path.basename(image_path)}",
                                   save=True,
                                   show=False)
    except KeyError:
        print("Unknown filter/no filter selected. No smoothing done to image.")
        smooth_image = image

    # x-axis convolution
    sobel_x = convolution(image=smooth_image,
                          kernel=kernels.SOBEL_KERNEL_X,
                          name=f"experiment{experiment}_sobel_x_{os.path.basename(image_path)}",
                          save=True,
                          show=False)

    # y-axis convolution
    sobel_y = convolution(image=smooth_image,
                          kernel=kernels.SOBEL_KERNEL_Y,
                          name=f"experiment{experiment}_sobel_y_{os.path.basename(image_path)}",
                          save=True,
                          show=False)

    # combine convolutions (edge strength)
    sobel_image = combine(sobel_x, sobel_y)
    cv2.imshow(f"experiment{experiment}_sobel_{os.path.basename(image_path)}", sobel_image)
    cv2.imwrite(f"{SAVE_PATH}experiment{experiment}_sobel_{os.path.basename(image_path)}", sobel_image)
    # histogram(sobel_image, f"{SAVE_PATH}experiment{experiment}_histogram_{os.path.basename(image_path)}")

    cv2.namedWindow(f"experiment{experiment}_threshold_{os.path.basename(image_path)}")
    cv2.createTrackbar('Threshold', f"experiment{experiment}_threshold_{os.path.basename(image_path)}", 80, 255,
                       lambda x: change(x=x, experiment=experiment, i=sobel_image))
    cv2.imshow(f"experiment{experiment}_threshold_{os.path.basename(image_path)}", sobel_image)
    change(i=sobel_image, experiment=experiment, x=80)
    # cv2.imwrite(f"{SAVE_PATH}experiment{experiment}_threshold_{os.path.basename(image_path)}", sobel_image)
    return sobel_image


def histogram(i, name="untitled"):
    hist = cv2.calcHist([i], [0], None, [256], [0, 256])
    hist = hist.reshape(256)

    # Plot histogram
    plt.bar(np.linspace(0, 255, 256), hist)
    plt.title(f'Histogram for {name}')
    plt.ylabel('Frequency')
    plt.xlabel('Grey Level')
    plt.show()
    plt.savefig(f'{name} Histogram.png')


# User can define image instead of sample image ("../kitty.bmp")
if len(sys.argv) == 1:
    image_path = SAMPLE_IMAGE
else:
    image_path = sys.argv[1]

img = open_image(image_path, f"{os.path.basename(image_path)}")

# experiment 0 - no smoothing used
# experiment 1 - using mean kernel
# experiment 2 - using a 3x3 gaussian
# experiment 3 - using a 5x5 gaussian
# experiment 4 - using a 7x7 gaussian
s = [sobel(img, experiment=0, mean="no_smoothing"),
     sobel(img, experiment=1, mean="mean"),
     sobel(img, experiment=2, mean="3x3"),
     sobel(img, experiment=3, mean="5x5"),
     sobel(img, experiment=4, mean="7x7")]

while True:
    if cv2.waitKey(1) == ord(' '):
        break

for e in range(len(s)):
    cv2.imwrite(f"{SAVE_PATH}experiment{e}_result_{os.path.basename(image_path)}", s[e])

cv2.destroyAllWindows()

bins = np.linspace(0, 255, 255)

plt.hist(cv2.calcHist([s[0]], [0], None, [256], [1, 255]), bins, label='original')
plt.hist(cv2.calcHist([s[2]], [0], None, [256], [1, 255]), bins, label='3x3 Gaussian')
plt.legend(loc='upper right')
plt.show()
plt.savefig(f'Histogram for comparison.png')
