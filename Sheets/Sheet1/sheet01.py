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
    diff = np.abs(np.subtract(im_1, im_2))
    max_error = np.max(diff)
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
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    # init 2D kernel filled with zeros
    kernel = np.zeros((kernel_size, kernel_size), dtype="float64")
    # iterate over 2 dimensions and compute a value for every pixel with gaussian distribution
    for i in range(-(kernel_size // 2), (kernel_size // 2) + 1):
        for j in range(-(kernel_size // 2), (kernel_size // 2) + 1):
            kernel[i + (kernel_size // 2), j + (kernel_size // 2)] = np.exp(-0.5 * (i * i + j * j) / (sigma * sigma))
    # return normalized kernel
    return kernel / kernel.sum()

def get_1D_kernel(sigma):
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    # init 1D kernel filled with zeros
    kernel = np.zeros((kernel_size), dtype="float64")
    # iterate over 1 dimension and compute a value for every pixel with gaussian distribution
    for i in range(-(kernel_size // 2), (kernel_size // 2) + 1):
        kernel[i + (kernel_size // 2)] = np.exp(-0.5 * (i * i) / (sigma * sigma))
    # return normalized kernel
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
    blur_img_a = cv.GaussianBlur(img_gray, (0, 0), sigma)
    plt.imshow(blur_img_a)
    plt.show()
    # 4 - b -
    blur_img_b = cv.filter2D(img_gray, -1, get_kernel(sigma))
    plt.imshow(blur_img_b)
    plt.show()

    #4 - c -
    one_d_kernel = get_1D_kernel(sigma=sigma)

    blur_img_c = cv.sepFilter2D(img_gray, -1, one_d_kernel, one_d_kernel)
    plt.imshow(blur_img_c)
    plt.show()

    # Compute the absolute pixel-wise difference between all pairs (there are three pairs) and print the maximum pixel
    # error for each pair.
    max_diff_1 = error(blur_img_a, blur_img_b)[1]
    max_diff_2 = error(blur_img_a, blur_img_c)[1]
    max_diff_3 = error(blur_img_c, blur_img_b)[1]
    print(f"Max Difference between cv.GaussianBlur and cv.filter2D results: {max_diff_1}")
    print(f"Max Difference between cv.GaussianBlur and cv.sepFilter2D results: {max_diff_2}")
    print(f"Max Difference between cv.filter2D and cv.sepFilter2D results: {max_diff_3}")

    print("============================================================\n")
# ************************************************
# ********************TASK5***********************
def task5():
    print("========================== Task 5 ==========================")
    # set image path
    img_path = 'bonn.png'
    # read img
    img = cv.imread(img_path)
    # convert to grey
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.imshow(img_gray)
    plt.show()

    height, width = img.shape[:2]
    sigma_a = 2
    sigma_b = 2 * np.sqrt(2)

    # Filter the image twice with a Gaussian kernel with sigma = 2
    img_a = cv.GaussianBlur(img_gray, (0, 0), sigma_a)
    img_a = cv.GaussianBlur(img_a, (0, 0), sigma_a)
    # Filter the image once with a Gaussian kernel with sigma = 2 * np.sqrt(2)
    img_b = cv.GaussianBlur(img_gray, (0, 0), sigma_b)
    # display both results
    plt.imshow(img_a)
    plt.show()
    plt.imshow(img_b)
    plt.show()
    #compute the absolute pixel-wise difference between the results, and print the maximum pixel error
    max_diff = error(img_a, img_b)[1]
    print(f"Max Difference : {max_diff}")
    print("============================================================\n")


# ************************************************
# ********************TASK7***********************
def add_salt_n_pepper_noise(img):
    img = img.astype(np.uint8)
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            # 30% probability of noize
            if np.random.uniform() < 0.3:
                # I assume that salt and pepper are equaly probable
                if np.random.uniform() < 0.5:
                    img[i, j] = 0
                else:
                    img[i, j] = 255
    return img

def mean_distance(img_1, img_2):
    return np.mean(np.abs(img_1 - img_2))



def task7():
    print("========================== Task 7 ==========================")
    # set image path
    img_path = 'bonn.png'
    # read img
    img = cv.imread(img_path)
    # convert to grey
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    noizy_img = add_salt_n_pepper_noise(img_gray)
    plt.imshow(noizy_img)
    plt.show()

    filter_size_arr = [1, 3, 5, 7, 9]
    filter_name_arr = ["Gaussian", "Median", "Bilateral"]
    # list to store best filtered imgs
    best_filtered_img_arr = [None, None, None]
    # list to store best mean distances, init with +inf
    best_distance_arr = [np.inf, np.inf, np.inf]
    best_filter_size_arr = [0, 0, 0]

    for filer_size in filter_size_arr:
        gauss_result = cv.GaussianBlur(noizy_img, (0, 0), filer_size)
        median_result = cv.medianBlur(noizy_img, filer_size)
        bilater_result = cv.bilateralFilter(noizy_img, -1, 1000.0, filer_size)
        result_arr = [gauss_result, median_result, bilater_result]
        distance_arr = [
                        mean_distance(gauss_result, img_gray),
                        mean_distance(median_result, img_gray),
                        mean_distance(bilater_result, img_gray)
                        ]
        # print(distance_arr)
        for k in range(3):
            if best_distance_arr[k] > distance_arr[k]:
                best_distance_arr[k] = distance_arr[k]
                best_filtered_img_arr[k] = result_arr[k]
                best_filter_size_arr[k] = filer_size

    for i in range(3):
        print(f"Best filter size for {filter_name_arr[i]} filter: {best_filter_size_arr[i]}")
        plt.imshow(best_filtered_img_arr[i])
        plt.show()







# ************************************************
# ********************TASK8***********************
def task8():
    # Your implementation of task 8
    pass

if __name__ == '__main__':
    # task1()
    # task2()
    # task4()
    # task5()
    task7()
