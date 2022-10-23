# Exercise Sheet 02

import cv2
import numpy as np
import os
import time

# blur the image in the spatial domain using convolution
def blur_im_spatial(image, kernel_sixse):
    #TODO
    

# blur the image in the frequency domain
def blur_im_freq(image, kernel):
    #TODO
    
# implement the ed s
def normalized_cross_correlation(image, template):
    # TODO


def task1(input_im_file):
    full_path = os.path.join('../data', input_im_file)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    kernel_siZe = 7
    kernel = None  # TODO: create kernel
    # time the blurring of the different methods
    start_time = time.time()
    conv_result = blur_im_spatial(image, kernel_siZe) 
    end_time = time.time()
    fft_result = blur_im_freq(image, kernel)

    # TODO: compare results in terms of run time and mean square difference





def task2():
    image = cv2.imread("../data/lena.png", 0)
    template = cv2.imread("../data/eye.png", 0)

    result_ncc = normalized_cross_correlation(image, template)

    # TODO: draw rectangle around found locations
    # TODO: show the results


def build_gaussian_pyramid_opencv(image, num_levels):
    # TODO: implement
    raise NotImplementedError


def build_gaussian_pyramid(image, num_levels):
    # TODO: implement
    raise NotImplementedError


def template_matching_multiple_scales(pyramid, template):
    # TODO: implement
    raise NotImplementedError


def task3():
    image = cv2.imread("../data/traffic.jpg", 0)
    template = cv2.imread("../data/template.jpg", 0)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)

    my_pyramid = build_gaussian_pyramid(image, 4)
    my_pyramid_template = build_gaussian_pyramid(template, 4)

    # TODO: compare and print mean absolute difference at each level

    # TODO: calculate the time needed for template matching without the pyramid

    result = template_matching_multiple_scales(my_pyramid, my_pyramid_template)
    # TODO: calculate the time needed for template matching with the pyramid

    # TODO: show the template matching results using the pyramid


def get_derivative_of_gaussian_kernel(size, sigma):
    # TODO: implement
    raise NotImplementedError


def task4():
    image = cv2.imread("../data/einstein.jpeg", 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = None  # TODO: convolve with kernel_x
    edges_y = None  # TODO: convolve with kernel_y

    magnitude = None  # TODO: compute edge magnitude
    direction = None  # TODO: compute edge direction

    cv2.imshow("Magnitude", magnitude)
    cv2.imshow("Direction", direction)


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    # TODO: implement
    raise NotImplementedError


def task5():
    image = cv2.imread("../data/traffic.jpg", 0)

    edges = None  # TODO: compute edges

    dist_transfom_mine = l2_distance_transform_2D()
    dist_transfom_cv = None  # TODO: compute using opencv

    # TODO: compare and print mean absolute difference


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()