import math
import sys
import cv2
import os
import numpy as np
import numpy.ma
import scipy
from matplotlib import pyplot as plt


def getImage(i):
    # Display Image
    image = cv2.imread(i)
    print(f"trying to open: {i}")

    if image is None:
        print(f'Error: failed to open {i}')
        sys.exit()

    PATH = f"./{os.path.dirname(os.path.relpath(i))}/"
    FILENAME = os.path.splitext(os.path.basename(i))[0]
    EXTENSION = os.path.splitext(os.path.basename(i))[1]

    print(f"Opened: {i}")

    print(f"Finding points for: {i}")
    image_points = harrisPointsDetector(image, PATH, FILENAME, EXTENSION)
    print(f"Found points for: {i}")

    print(f"Finding features for: {i}")
    image_descriptors, image_features = featureDescription(image, image_points, PATH, FILENAME, EXTENSION)
    print(f"Found features for: {i}")

    return {"image": image,
            "points": image_points,
            "features": image_features,
            "descriptors": image_descriptors,
            "path": PATH,
            "filename": FILENAME,
            "ext": EXTENSION}


def sobel(image, PATH, FILENAME, EXTENSION):
    # gradient_x
    grad_x = scipy.ndimage.sobel(image, axis=0, mode="reflect")

    # gradient_y
    grad_y = scipy.ndimage.sobel(image, axis=1, mode="reflect")

    cv2.imwrite(f"{PATH}{RESULTS_DIR}sobel_x_{FILENAME}{EXTENSION}", grad_x)
    cv2.imwrite(f"{PATH}{RESULTS_DIR}sobel_y_{FILENAME}{EXTENSION}", grad_y)

    return [grad_x, grad_y]


def interestPoint(harris, PATH, FILENAME, EXTENSION):
    # threshold
    local_maxima = scipy.ndimage.maximum_filter(harris,
                                                size=(7, 7),
                                                mode="reflect")

    threshold_value = np.percentile(local_maxima, 99.9)

    v, local_maxima = cv2.threshold(local_maxima,
                                    threshold_value,
                                    255,
                                    cv2.THRESH_BINARY)  # threshold value 5000000

    thresholds[0].append(v)
    thresholds[1].append(len(local_maxima))

    # non-local maxima suppression
    for x_index, row in enumerate(local_maxima):
        for y_index, point in enumerate(row):
            try:
                if int(local_maxima[x_index][y_index]) >= 255:
                    # create area around point
                    left = x_index - 3 if x_index >= 3 else 0
                    top = y_index - 3 if y_index >= 3 else 0
                    bottom = y_index + 3
                    right = x_index + 3
                    for i in range(left, right):
                        for j in range(top, bottom):
                            # remove neighbouring points
                            if int(local_maxima[i][j]) >= 255 \
                                    and i != x_index \
                                    and j != y_index:
                                local_maxima[i][j] = 0.0
            except IndexError:
                continue

    return local_maxima


def harrisPointsDetector(image, PATH, FILENAME, EXTENSION):
    # convert to greyscale
    grey_image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=float)
    cv2.imwrite(f"{PATH}{RESULTS_DIR}greyscale_{FILENAME}{EXTENSION}", grey_image)

    # blur image to remove noise
    blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 0.5)
    cv2.imwrite(f"{PATH}{RESULTS_DIR}blurred_{FILENAME}{EXTENSION}", blurred_image)

    # Spatial derivative calculation (sobel edge detection) -> returns an array [x, y]
    sobel_images = sobel(blurred_image, PATH, FILENAME, EXTENSION)

    # eigenvalues - square of derivatives & gaussian filter
    Ixx = scipy.ndimage.gaussian_filter(sobel_images[0] ** 2,
                                        sigma=0.5,
                                        mode="reflect",
                                        radius=2)

    Iyy = scipy.ndimage.gaussian_filter(sobel_images[1] ** 2,
                                        sigma=0.5,
                                        mode="reflect",
                                        radius=2)

    Ixy = scipy.ndimage.gaussian_filter(sobel_images[0] * sobel_images[1],
                                        sigma=0.5,
                                        mode="reflect",
                                        radius=2)

    # Harris response
    determinant = (Ixx * Iyy) - (np.square(Ixy))
    trace = Ixx + Iyy

    r = determinant - (0.05 * trace ** 2)

    # supress non-local maxima
    maxima = interestPoint(r, PATH, FILENAME, EXTENSION)

    points = []

    # turn points into key points
    for row_index, row in enumerate(maxima):
        for column_index, p in enumerate(row):

            if p != 0:
                points.append(cv2.KeyPoint(x=column_index, y=row_index, size=1.0))

    return points


def featureDescription(image, points, PATH, FILENAME, EXTENSION):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # compute the descriptors with ORB
    kp, des = orb.compute(image, points)

    # draw only key points location, not size and orientation
    new_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)

    cv2.imwrite(f"{PATH}{RESULTS_DIR}feature_description_{FILENAME}{EXTENSION}", new_image)
    cv2.imshow(f"feature_description_{FILENAME}", new_image)

    return np.array(des), np.array(kp)


def ssd(image1, image2):
    print("SSD")

    best_matches = [cv2.DMatch() for _ in range(len(image1['descriptors']))]
    distance = scipy.spatial.distance.cdist(image1["descriptors"], image2["descriptors"])

    for index, val in enumerate(distance):

        best_matches[index].queryIdx = index
        best_matches[index].trainIdx = numpy.ma.argmin(val)
        best_matches[index].distance = val[numpy.ma.argmin(val)]

    return best_matches


def ratio(image1, image2):
    print("ratio")

    best_matches = [cv2.DMatch() for _ in range(len(image1['descriptors']))]
    distance = scipy.spatial.distance.cdist(image1["descriptors"], image2["descriptors"])

    for index, val in enumerate(distance):

        smallest_index = numpy.ma.argmin(val)
        smallest = val[smallest_index]
        temp = val.copy()
        numpy.delete(temp, smallest_index)
        second_smallest_index = numpy.ma.argmin(temp)
        second_smallest = temp[second_smallest_index]

        best_matches[index].queryIdx = index
        best_matches[index].trainIdx = numpy.ma.argmin(val)
        best_matches[index].distance = smallest / second_smallest

    return best_matches


def matchFeatures(image1, image2, matching=ratio):
    print(f"feature matching: {image1['filename']} and {image2['filename']}")

    match = matching(image1, image2)
    match = [valid for valid in match if not (math.isnan(valid.distance))]
    match.sort(key=lambda x: x.distance)

    # shows first 50 matches
    matches_image = cv2.drawMatches(image1['image'],
                                    image1['features'],
                                    image2['image'],
                                    image2['features'],
                                    match[:50],
                                    None)

    cv2.imwrite(f"{image1['path']}{RESULTS_DIR}{image1['filename']}_matching_{image2['filename']}.{image1['ext']}",
                matches_image)

    return matches_image


def histogram(i, name="./untitled.jpg"):
    hist = cv2.calcHist([i], [0], None, [256], [0, 256])
    hist = hist.reshape(256)

    # Plot histogram
    plt.bar(np.linspace(0, 255, 256), hist)
    plt.title(f'Histogram for {name}')
    plt.ylabel('Frequency')
    plt.xlabel('Grey Level')
    plt.show()
    plt.savefig(f'{name}')


SAMPLE_IMAGE = "./images/bernieSanders.jpg"
RESULTS_DIR = "/results/"
K = 0.05

if len(sys.argv) == 1:
    image_1 = SAMPLE_IMAGE
    img1 = getImage(image_1)
    sys.exit()

elif len(sys.argv) >= 3:
    thresholds = [[], []]
    test_images = []
    img1 = getImage(sys.argv[1])
    for im in range(2, len(sys.argv)):
        test_images.append(getImage(sys.argv[im]))

else:
    image_1 = sys.argv[1]
    img1 = getImage(image_1)
    sys.exit()

for img in test_images:
    m = matchFeatures(img1, img)

plt.plot(thresholds[0], thresholds[1], 'o')
plt.xlabel("Threshold Value")
plt.ylabel("Number of Interest Points")
plt.show()

cv2.destroyAllWindows()
