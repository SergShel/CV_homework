import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint

authors = ["Siarhei Sheludzko", "Marcel Melchers"]


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    # set image path
    img_path = 'bonn.png'

    # 2a: read and display the image
    img = cv.imread(img_path)
    if img is None:
        sys.exit("Could not read the image.")
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation
    img_cpy = img.copy()
    height, width, channels = img_cpy.shape

    # Going through the pixels
    for x in range(height):
        for y in range(width):
            for c in range(channels):
                img_cpy[x, y, c] = max(0, img_cpy[x, y, c] - 0.5 * img_gray[x, y])

    display_image('2 - c - Reduced Intensity Image', img_cpy)


    # 2d: one-line statement to perfom the operation above
    """
    * np.expand_dims add channel dim to the img-grey
    * clip(0, 255) limit the values of the pixels
    * set dtype to uint8
    """
    img_cpy = (img - np.expand_dims(0.5 * img_gray, axis=2)).clip(min=0, max=255).astype(np.uint8)
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)

    # 2e: Extract the center patch and place randomly in the image
    center_m = img.shape[0] // 2
    center_n = img.shape[1] // 2

    patch = img[center_m - 8:center_m + 8, center_n - 8:center_n + 8, :]

    # prevent overlapping and shape errors
    # center_m = np.random.random_integers(16, img.shape[0] - 17)
    # center_n = np.random.random_integers(16, img.shape[1] - 17)

    # img_patch = img.copy()
    # img_patch[center_m - 8:center_m + 8, center_n - 8:center_n + 8] = patch

    display_image('2 - e - Center Patch', patch)

    # Random location of the patch for placement
    rand_coord = [np.random.randint(0, height-17), np.random.randint(0, width-17)]
    img_cpy = img.copy()
    img_cpy[rand_coord[0]:rand_coord[0]+16, rand_coord[1]:rand_coord[1]+16] = patch
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)

    # 2f: Draw random rectangles and ellipses
    img_cpy = img.copy()

    num_figures = 10 # 10 rectangles and ellipses
    # random x, y, size_x, size_y, (R,G,B) for rectangles
    random_rectangle_values = np.array([
        randint(0, width, num_figures), randint(0, height, num_figures),
        randint(10, 70, num_figures), randint(10, 90, num_figures),
        randint(0, 255, num_figures), randint(0, 255, num_figures), randint(0, 255, num_figures)
    ]).astype(np.uint16)
    # random x, y, (R,G,B), angle for ellipses
    random_ellipse_values = np.array([
        randint(0, width, num_figures), randint(0, height, num_figures),
        randint(0, 255, num_figures), randint(0, 255, num_figures), randint(0, 255, num_figures),
        randint(0, 360, num_figures)
    ]).astype(np.uint16)

    for i in range(num_figures):
        rect_x = random_rectangle_values[0, i]
        rect_y = random_rectangle_values[1, i]

        rect_other_corner = (rect_x + random_rectangle_values[2, i], rect_y + random_rectangle_values[3, i])
        rect_color = (int(random_rectangle_values[4, i]), int(random_rectangle_values[5, i]), int(random_rectangle_values[6, i]))
        # add rectangle
        cv.rectangle(img_cpy, (rect_x, rect_y), rect_other_corner, rect_color, -1)

        ellipse_x = random_ellipse_values[0, i]
        ellipse_y = random_ellipse_values[1, i]

        ellipse_color = (int(random_ellipse_values[2, i]), int(random_ellipse_values[3, i]), int(random_ellipse_values[4, i]))
        # add ellipse
        cv.ellipse(img_cpy, (ellipse_x, ellipse_x), (15, 24), random_ellipse_values[5, i], 0, 360, ellipse_color, -1)

    display_image('2 - f - Rectangles and Ellipses', img_cpy)

    # destroy all windows
    cv.destroyAllWindows()
