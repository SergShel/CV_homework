import cv2 as cv
import numpy as np
import random
import time
import matplotlib.pyplot as plt


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# ********************TASK1***********************
def integral_image(img):
    # get img height and width
    height, width = img.shape
    # initialize integral matrix with shape of img +1 col and +1 row with zeros
    integral_img = np.zeros((height+1, width+1), np.uint64)
    # iterate over all pixels in img with 2 for-loops
    for y in range(1, height+1):
        for x in range(1, width+1):
            integral_img[y, x] = integral_img[y-1, x] + integral_img[y, x-1] - integral_img[y-1, x-1] + img[y-1, x-1]
    return integral_img[1:, 1:]


def sum_image(image):
    height, width = image.shape
    sum_val = 0
    for y in range(height):
        for x in range(width):
            sum_val += image[y, x]
    return sum_val


def task1():
    # set image path
    img_path = 'bonn.png'
    # read img
    img = cv.imread(img_path)
    height, width = img.shape[:2]
    # convert to grey
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 1 - a calculate integral img
    integral_img = integral_image(img_gray)
    plt.imshow(integral_img)
    plt.show()
    # display_image('1 - a - Integral Image', (integral_img / integral_img[-1, -1]))

    # 1 - b (i)
    mean_grey_1 = sum_image(img_gray) / (height * width)
    print("Mean grey I  : ", mean_grey_1)
    # 1 - b (ii)
    mean_grey_2 = cv.integral(img_gray)[-1, -1] / (height * width)
    print("Mean grey III: ", mean_grey_2)
    # 1 - b (iii)
    mean_grey_3 = integral_image(img_gray)[-1, -1] / (height * width)
    print("Mean grey II : ", mean_grey_3)
    print("\n")

    # 1 - c
    sum_time = 0
    integr_cv_time = 0
    integr_my_time = 0
    for i in range(10):
        # generate random coords for top left corner of 100x100-patch
        top_left_x = np.random.randint(0, width-100)
        top_left_y = np.random.randint(0, height-100)
        # first method
        start = time.time()
        mean_grey_1_i = sum_image(img_gray[top_left_y: (top_left_y + 100), top_left_x: (top_left_x + 100)]) / (100 * 100)
        sum_time += time.time() - start

        # second method
        np_integr_img = cv.integral(img_gray)
        start = time.time()
        mean_grey_2_i = (np_integr_img[top_left_y + 100, top_left_x + 100] +
                        np_integr_img[top_left_y, top_left_x] -
                        np_integr_img[top_left_y, top_left_x + 100] -
                        np_integr_img[top_left_y + 100, top_left_x]) / (100 * 100)
        integr_cv_time += time.time() - start

        # third method
        start = time.time()
        mean_grey_3_i = (integral_img[top_left_y + 100, top_left_x + 100] +
                        integral_img[top_left_y, top_left_x] -
                        integral_img[top_left_y, top_left_x + 100] -
                        integral_img[top_left_y + 100, top_left_x]) / (100 * 100)
        integr_my_time += time.time() - start

    print(f'Total time taken by sum:           {sum_time}')
    print(f'Total time taken by integral (cv): {integr_cv_time}')
    print(f'Total time taken by integral (my): {integr_my_time}')







# ************************************************
# ********************TASK2***********************
def equalize_hist_image(img):
    # Your implementation of histogram equalization
    pass


def task2():
    # Your implementation of Task2
    pass
# ************************************************
# ********************TASK4***********************
def get_kernel(sigma):
    # Your implementation of getGaussianKernel
    pass


def task4():
    # Your implementation of Task4
    pass
# ************************************************
# ********************TASK5***********************
def task5():
    # Your implementation of Task5
    pass
# ************************************************
# ********************TASK7***********************
def add_salt_n_pepper_noise(img):
    # Your implementation of adding noise to the image
    pass


def task7():
    # Your implementation of task 7
    pass

# ************************************************
# ********************TASK8***********************
def task8():
    # Your implementation of task 8
    pass

if __name__ == '__main__':
    task1()
