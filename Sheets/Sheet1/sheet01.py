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
    """
    Calculates unnormalized integral image
    :param img: intensity image
    :return: unnormalized integral image
    """
    integral_image = np.int32(cv.copyMakeBorder(img, top=1, bottom=0, left=1, right=0, borderType=cv.BORDER_CONSTANT, value=0))
    height, width = img.shape[:2]
    for y in range(height):
        for x in range(width):
            integral_image[y+1, x+1] = img[y, x] + integral_image[y+1, x] + integral_image[y, x+1] - integral_image[y, x]
    return integral_image


def sum_image(image):
    height, width = image.shape
    sum_val = 0
    for y in range(height):
        for x in range(width):
            sum_val += image[y, x]
    return sum_val


def task1():
    print("========================== Task 1 ==========================")

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

    # Task1_b
    def dummy_mean(img):
        pixel_num = img.shape[0] * img.shape[1]
        return sum_image(img) / pixel_num

    def my_integral_mean(img):
        pixel_num = img.shape[0] * img.shape[1]
        integral_img = integral_image(img)
        bottom_right = integral_img[-1, -1]
        top_left = integral_img[0, 0]
        top_right = integral_img[0, -1]
        bottom_left = integral_img[-1, 0]
        return (bottom_right + top_left - bottom_left - top_right) / pixel_num

    def integral_mean(img):
        pixel_num = img.shape[0] * img.shape[1]
        integral_img = cv.integral(img)
        bottom_right = integral_img[-1, -1]
        top_left = integral_img[0, 0]
        top_right = integral_img[0, -1]
        bottom_left = integral_img[-1, 0]
        return (bottom_right + top_left - bottom_left - top_right) / pixel_num


    # 1 - b (i)
    mean_grey_1 = dummy_mean(img_gray)
    print("Mean grey I  : ", mean_grey_1)
    # 1 - b (ii)
    mean_grey_2 = integral_mean(img_gray)
    print("Mean grey III: ", mean_grey_2)
    # 1 - b (iii)
    mean_grey_3 = my_integral_mean(img_gray)
    print("Mean grey II : ", mean_grey_3)
    print("\n")

    # Task1_c

    def benchmark(image, mean_function, function_name, random_coordinates):
        start = time.perf_counter()
        for coordinate in random_coordinates:
            square_patch = image[coordinate[0]: coordinate[0] + 100, coordinate[1]: coordinate[1] + 100]
            # display_image("Test 100x100 patches", square_patch)  # uncomment the line to check the patches
            mean = mean_function(square_patch)

        print(f"Runtime of the task with <{function_name}>: {time.perf_counter() - start} seconds")

    # generate 10 2D-coordinates of upper-left corner of 100x100 patches
    random_coords = np.random.randint(low=[0, 0], high=[img.shape[0] - 100, img.shape[1] - 100], size=(10, 2))

    # Bechmarking:
    benchmark(image=img_gray, mean_function=dummy_mean, function_name="Nested for-loops mean",
              random_coordinates=random_coords)
    benchmark(image=img_gray, mean_function=my_integral_mean, function_name="My Integral mean",
              random_coordinates=random_coords)
    benchmark(image=img_gray, mean_function=integral_mean, function_name="CV2 Integral mean",
              random_coordinates=random_coords)

    print("===========================================================\n")





# ************************************************
# ********************TASK2***********************
def equalize_hist_image(img):
    # get image shape
    height, width = img.shape[:2]
    # create empty image filled with zeros
    equalized_img = np.zeros((height, width), np.uint64)
    # Count the frequency of intensities
    hist, bins = np.histogram(img.flatten(), 256, [0, 256], density=True)
    # Compute integral histogram/CDF - representing the new intensity values
    hist = np.cumsum(hist)
    # Fill the new image -> replace old intensity values with new intensities taken from the integral histogram
    for y in range(height):
        for x in range(width):
            equalized_img[y, x] = int(hist[img[y, x]] * 255.)

    return equalized_img.astype(np.uint8)

def error(im_1,im_2):
    diff = cv.absdiff(im_1.astype("uint8"), im_2.astype("uint8"))
    max_error = np.amax(diff)
    return diff, max_error


def task2():
    print("========================== Task 2 ==========================")
    # set image path
    img_path = 'bonn.png'
    # read img
    img = cv.imread(img_path)
    height, width = img.shape[:2]
    # convert to grey
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    equalized_img = cv.equalizeHist(img_gray)

    plt.imshow(equalized_img)
    plt.show()
    display_image('2 - a - Image with the histogram equalization', equalized_img)

    my_equalized_img = equalize_hist_image(img_gray)

    plt.imshow(my_equalized_img)
    plt.show()

    diff_img = img_gray - equalized_img
    plt.imshow(diff_img)
    plt.show()

    abs_error, max_error = error(my_equalized_img, img_gray)
    print("abs Error: \n" + str(abs_error) + "\n max Error: " + str(max_error))

    # display_image('bonn grey', img_gray)
    # display_image('bonn_equalizeHist', my_equalized_img)

    print("===========================================================\n")


# ************************************************
# ********************TASK4***********************
def get_kernel(sigma):

    kernel_size = int(np.ceil(4 * sigma) + 1)
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    print(kernel_size)
    kernel = np.zeros((kernel_size, kernel_size), dtype="float64")

    for i in range(-(kernel_size // 2), (kernel_size // 2) + 1):
        for j in range(-(kernel_size // 2), (kernel_size // 2) + 1):
            kernel[i + (kernel_size // 2), j + (kernel_size // 2)] = np.exp(-0.5 * (i * i + j * j) / (sigma * sigma))
    # return normalized kernel
    return kernel / kernel.sum()

def get_1D_kernel(sigma):
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = np.zeros((kernel_size), dtype="float64")
    for i in range(-(kernel_size // 2), (kernel_size // 2) + 1):
        kernel[i + (kernel_size // 2)] = np.exp(-0.5 * (i * i) / (sigma * sigma))
    return kernel / kernel.sum()


def task4():
    print("========================== Task 4 ==========================")
    # set image path
    img_path = 'bonn.png'
    # read img
    img = cv.imread(img_path)
    sigma = 2 * np.sqrt(2)
    height, width = img.shape[:2]
    # convert to grey
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 4 - a - GaussianBlur
    blur_img_a = cv.GaussianBlur(img_gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    plt.imshow(blur_img_a)
    plt.show()
    # 4 - b -
    blur_img_b = cv.filter2D(img_gray, -1, get_kernel(sigma))
    plt.imshow(blur_img_b)
    plt.show()

    #4 - c -
    one_d_kernel = get_1D_kernel(sigma=sigma)
    print("my kernel:", one_d_kernel)

    blur_img_c = cv.sepFilter2D(img_gray, -1, one_d_kernel, one_d_kernel)
    plt.imshow(blur_img_c)
    plt.show()

    print("============================================================\n")
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
    # task1()
    # task2()
    task4()
