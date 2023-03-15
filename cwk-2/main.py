import sys
import cv2
import os
import numpy as np

SAMPLE_IMAGE = "./images/bernieSanders.jpg"


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


def harrisPointsDetector(image):
    # TO BE REMOVED FROM CODE
    grey = np.float32(image)
    dst = cv2.cornerHarris(grey, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    grey[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('dst', image)
    return dst


# User can define image instead of sample image ("../kitty.bmp")
if len(sys.argv) == 1:
    image_path = SAMPLE_IMAGE
else:
    image_path = sys.argv[1]

# open image of bernie sanders
img = open_image(image_path, f"{os.path.basename(image_path)}")
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_points = harrisPointsDetector(img)

while True:
    if cv2.waitKey(1) == ord(' '):
        break

cv2.destroyAllWindows()
