import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1  # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0  # Map is fixed point int with 4 fractional bits

    return disparity  # floating point image


# ================================================


# ================================================
#
def plot(disparity):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    x = []
    y = []
    z = []
    print("plotting disparity")
    for index, val in enumerate(disparity):
        x += [val[0]]
        y += [val[1]]
        z += [val[2]]

    # for r in range(4):
    #     for c in range(4):
    #         x += [c]
    #         y += [r]
    #         z += [r * c]

    # Plt depths
    ax = plt.axes(projection='3d')
    ax.scatter(x, z, y, c='green', s=0.2)

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # ax.view_init(vertical_axis="y", roll=0, elev=20, azim=-40)
    # ax.view_init(30, 170, vertical_axis='y')
    # plt.savefig('./img/output/3D_view.png', bbox_inches='tight')  # Can also specify an image, e.g. myplot.png
    # plt.show()
    ax.view_init(90, -90)
    plt.savefig('./img/output/xz_view.png', bbox_inches='tight')  # Can also specify an image, e.g. myplot.png
    plt.show()
    # ax.view_init(0, 180)
    # plt.savefig('./img/output/yz_view.png', bbox_inches='tight')  # Can also specify an image, e.g. myplot.png
    # plt.show()
    print("disparity plotted")


# ================================================
#
if __name__ == '__main__':

    # Load left image
    filename = 'img/input/umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'img/input/umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 64, 5)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)

    # Show 3D plot of the scene
    plot(disparity)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
