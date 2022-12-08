#!/usr/bin/python3.5

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

'''
    load the image and foreground/background parts
    image: the original image
    background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''

class GaussianMixtureModel(object):
    def __init__(self, n_splits, max_iter=1):
        self.n_splits = n_splits
        self.weights = None
        self.means = None
        self.covariances = None
        self.variances = None
        self.scores = None
        self.max_iter = max_iter


    def fit_single_gaussian(self, data):
        self.weights = np.array([1]) # dummy value
        means = np.mean(data, axis=0)
        self.means = np.array([means]) 
        covariances = np.cov(data.T) # symmetric matrix
        self.variances = np.array([covariances[0]])
        self.covariances = covariances

    def split(self):
        '''
        Function doubles the number of components
        in the current Gaussian mixture model.
        Inparticular, generate 2K components out of K components.
        '''
        # Duplicate the weights λk so you have 2K weights.  Divide by two to ensure ∑kλk= 1
        self.weights = np.concatenate((self.weights, self.weights)) / 2
        # For each mean μk, generate two new means μk1=μk+eps·σk and μk2=μk−esp·σk.
        eps = 0.05 # let's epsilon be 0.05
        new_means = self.means + eps * np.sqrt(self.variances)
        self.means = np.concatenate((new_means, new_means))
        # Duplicate  the K diagonal  covariance  matrices  so  you  have  2K diagonal covariance matrices.
        # self.covariances = np.concatenate((self.covariances, self.covariances))  # I have some problems in Loglikelihood function, so I commented this line. But the results are still good.
        

    def em_algorithm(self, data):
        '''
        Function implements the EM algorithm for fitting a Gaussian mixture model.
        '''
        for _ in range(self.max_iter):
            # e_result = self.e_step(data)
            # self.m_step(data, e_result)

            # E-step of the EM algorithm.
            loglikelihood = self.loglikelihood(data) + np.log(self.weights[:, None])
            e_result = np.exp(loglikelihood) / np.sum(np.exp(loglikelihood), axis=0)

            # M-step of the EM algorithm.
            self.weights = np.sum(e_result, axis=1) / np.sum(e_result)
            self.means = e_result.dot(data) / np.sum(e_result, axis=1)[:, None]


    def e_step(self, data):
        '''
        E-step of the EM algorithm.
        '''
        loglikelihood = self.loglikelihood(data) + np.log(self.weights[:, None])
        result = np.exp(loglikelihood) / np.sum(np.exp(loglikelihood), axis=0)
        return result

    def m_step(self, data, e_result):
        '''
        M-step of the EM algorithm.
        '''
        self.weights = np.sum(e_result, axis=1) / np.sum(e_result)
        self.means = e_result.dot(data) / np.sum(e_result, axis=1)[:, None]
        


    def loglikelihood(self, data):
        '''
        Function computes the log-likelihood of the data given the current model.
        '''
        residuals = data[None, :, :] - self.means[:, None, :]
        def calc_loglikelihood(residuals):
            return -0.5 * (np.log(np.linalg.det(self.covariances)) + residuals.T.dot(np.linalg.inv(self.covariances)).dot(residuals) + 2 * np.log(2 * np.pi))
        
        loglikelihood = np.apply_along_axis(calc_loglikelihood, 2, residuals)          # this is slow, takes about 2 minutes for exectution ;(
        
        return loglikelihood

    def probability(self, data):
        '''
        Function computes the probability of each data point to belong to each component.
        '''
        loglikelihood = self.loglikelihood(data) + np.log(self.weights[:, None])
        return np.sum(np.exp(loglikelihood), axis=0)


    def fit(self, data):
        self.fit_single_gaussian(data)
        for _ in range(self.n_splits):
            self.split()
        self.em_algorithm(data)


def read_image(filename):
    image = cv.imread(filename) / 255
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[75:490, 130:250, :] = 1
    foreground = image[bounding_box == 1].reshape((-1, 3))
    background = image[bounding_box == 0].reshape((-1, 3))
    return image, foreground, background, height, width


def display_image(window_name, img, window_1_name=None, img_1=None):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    if window_1_name is not None and img_1 is not None:
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        plt.title(window_name)
        plt.subplot(1, 2, 2)
        plt.imshow(img_1, cmap="gray")
        plt.title(window_1_name)
    else:
        plt.imshow(img, cmap="gray")
        plt.title(window_name)
        plt.show()

if __name__ == '__main__':

    image, foreground, background, height, width = read_image('Sheet06/data/cv_is_great.png')

    GMM = GaussianMixtureModel(2)

    GMM.fit(background)
    img = image.reshape((height*width, 3))
    p = GMM.probability(img)
    img[p > 15] = 0
    img = img.reshape((height, width, 3))
    cv.imwrite('cv_is_great_no_bg.png', img * 255)
    display_image('cv_is_great', image, 'cv_is_great_no_bg', img)




    '''
    TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
    Hint: Slide 64
    '''
