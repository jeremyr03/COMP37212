import sys
import cv2
import os
import numpy as np
import scipy
from matplotlib import pyplot as plt

PATH = "./images/"
EXTENSION = "jpg"
SAMPLE_IMAGE = f"{PATH}bernieSanders.{EXTENSION}"
# SAMPLE_IMAGE = f"./box.{EXTENSION}"
K = 0.05


def open_image(i, filename=None, name=None):
    # Display Image
    if not filename:
        filename = name if name else "untitled"
        image = i
    else:
        image = cv2.imread(i)

    if image is None:
        print(f'Error: failed to open {i}')
        sys.exit()
    # cv2.imwrite(f"{PATH}{filename}.{EXTENSION}", image)

    return image


def sobel(image):
    # gradient_x
    grad_x = cv2.Sobel(image,
                       cv2.CV_16S,
                       2,  # derivative for x
                       0,  # derivative for y
                       ksize=3,
                       scale=1,
                       delta=0,
                       borderType=cv2.BORDER_DEFAULT)
    # grad_x = scipy.ndimage.sobel(image, axis=0)

    # gradient_y
    grad_y = cv2.Sobel(image,
                       cv2.CV_16S,
                       0,  # derivative for x
                       2,  # derivative for y
                       ksize=3,
                       scale=1,
                       delta=0,
                       borderType=cv2.BORDER_DEFAULT)
    # grad_y = scipy.ndimage.sobel(image, axis=1)

    grad_xy = cv2.Sobel(image,
                        cv2.CV_16S,
                        1,  # derivative for x
                        1,  # derivative for y
                        ksize=3,
                        scale=1,
                        delta=0,
                        borderType=cv2.BORDER_DEFAULT)

    cv2.imwrite(f"{PATH}sobel_x_{FILENAME}.{EXTENSION}", grad_x)
    cv2.imwrite(f"{PATH}sobel_y_{FILENAME}.{EXTENSION}", grad_y)
    cv2.imwrite(f"{PATH}sobel_xy_{FILENAME}.{EXTENSION}", grad_xy)

    # returns [xx, yy, xy]
    return [grad_x, grad_y, grad_xy]


def harrisPointsDetector(image):
    # convert to greyscale
    grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{PATH}greyscale_{FILENAME}.{EXTENSION}", grey_image)

    # blur image to remove noise
    blurred_image = cv2.blur(grey_image, (5, 5))
    cv2.imwrite(f"{PATH}blurred_{FILENAME}.{EXTENSION}", blurred_image)

    # sobel edge detection -> returns dictionary of 'x' and 'y'
    sobel_images = sobel(blurred_image)

    # gaussian filter on each sobel image
    # 5x5 gaussian mask with 0.5 sigma
    # [xx, yy, xy]
    gradients = [cv2.GaussianBlur(i,
                                  ksize=(5, 5),  # 5x5 kernel
                                  sigmaX=0.5,  # sigma value
                                  sigmaY=0.5,
                                  borderType=cv2.BORDER_REFLECT) for i in sobel_images]
    # gradients = [scipy.ndimage.gaussian_filter(i,
    #                                            sigma=0.5,
    #                                            radius=2, mode="reflect") for i in sobel_images]

    # determinant and trace
    determinant = (gradients[0] * gradients[1]) - (gradients[2] ** 2)
    trace = gradients[0] + gradients[1]

    harris_response = determinant - 0.05 * trace ** 2

    corners = np.copy(image)
    edges = np.copy(image)

    # supress non-local maxima ?
    local_maxima = scipy.ndimage.maximum_filter(harris_response,
                                                size=(7, 7),
                                                mode="reflect")

    for rowindex, response in enumerate(harris_response):
        for colindex, r in enumerate(response):
            # corner
            if r > 0:
                # this is a corner
                corners[rowindex, colindex] = [0, 0, 255]
            # edge
            elif r < 0:
                # this is an edge
                edges[rowindex, colindex] = [0, 255, 0]

    cv2.imwrite("corners.jpg", corners)
    cv2.imwrite("edges.jpg", edges)

    # # TO BE REMOVED FROM CODE
    # heatmap = cv2.resize(heatmap, (400, 300))
    # plt.matshow(heatmap)
    # plt.show()
    # heatmapshow = None
    # heatmapshow = cv2.normalize(heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    # cv2.imshow("Heatmap", heatmapshow)

    return 0


# User can define image instead of sample image ("../kitty.bmp")
if len(sys.argv) == 1:
    image_path = SAMPLE_IMAGE
else:
    image_path = sys.argv[1]

# open image of bernie sanders
FILENAME = os.path.splitext(os.path.basename(image_path))[0]
img = open_image(image_path, filename=f"{FILENAME}")
img_points = harrisPointsDetector(img)

# while True:
#     if cv2.waitKey(1) == ord(' '):
#         break

cv2.destroyAllWindows()
