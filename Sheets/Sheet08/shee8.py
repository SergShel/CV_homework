import cv2
import numpy as np
import os
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg

DATA_DIR = '../data/task%s'

#
def decomposePCA(dataPoints,  k=None, preservRatio=None):
    # implement PCA for task1 yourself and return the first k 
    # components that preserve preservRatio of the energy and their eigen valuess
    pass

def loadData(filename):
    #return the data points
    df = pd.read_csv(filename, header = 0)
    pass

def visualiseHands(kpts, title):
    #use matplotlib for that
    pass


#task 1: training the statistical shape model
# return the trained model so you can use it in task 2
def task1(train_file='hands_aligned_train.txt'):

    trainfilePath = os.path.join(DATA_DIR%'1', train_file)
    




#task2: performing inference on the test hand data using the trained shape model
def task2(shapeModel, test_file='hands_aligned_test.txt'):
    testfilePath = os.path.join(DATA_DIR%'1', test_file)

    
    

#eigen faces
def task3():
    #(a)you can use scikit PCA for the decomposition and the dataset class for loading LFW dataset directly
    # train over LFW training split that you created
    #(b) evaluate over the samples in data/task3/detect/face and data/task3/detect/notFace
    
    positiveFilenames = os.listdir(DATA_DIR%'3', 'detect/face')
    negativeFilenames = os.path.listdir(DATA_DIR%'3', 'detect/notFace')
    #calculate the accuracy
    
    #(c) test over LFW test split and calculate the accuracy of recognition
    pass

#compute the structural tensor M, you can apply an opencv filters to calculate the gradients
def computeStructural(image):
    M = None
    #todo
    return M
def detectorCornerHarris(image, M, responseThresh):
    pass

def detectorCornerFoerstner(image, M, responseThresh):
    pass

# corner detectors: implement Harris corner detector and Förstner corner  
def task4(imFile ='palace.jpeg'):
    image = cv2.imread(DATA_DIR%'4', imFile)
    
    #(a)
    #todo
    #(b) apply Harris corner detector and visualise
    #todo
    #(c) apply Foerstner corner detector and visualise
    #todo
    pass

# perform the matching using sift
def match(sift, image1, image2):
    pass
def rotateImage(image, rotAngle):
    # to get the rotation matrix, you can use cv2.getRotationMatrix2D
    rotMat = None
    rotatedIm = cv2.warpAffine(image, rotMat) #fill in the missing parameters
    return rotatedIm

#keypoint matching
def task5(imFile = 'castle.jpeg'):
    
    
    image = cv2.imread(DATA_DIR%'5'%imFile)
    #you can use cv2's implementation of SIFT
    sift = cv2.SIFT_create()
    #apply each of the following transformations and then perform matching
    # Please choose two values of your choice for each one
    rotAngles = [0, 0] #todo choose two values
    for rotAngle in rotAngles:
        rotatedIm = rotateImage(image, rotAngle)
        
        match(sift, image, rotatedIm)
        #todo visualise using cv2.drawMatches
    
    #do the same with a translation transformation
    
    #do the same with a scaling transformation
 
#Image homography: implement the RANSAC algorithm then apply it to stitch the two images of Bonn's
# Poppelsdorfer Schloss and then visualise the stitched image


def task6():


    # Read the two images
    image1 = cv2.imread("./data/task6/schloss1.jpeg")
    image2 = cv2.imread('./data/task6/schloss2.jpeg')
    
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute their descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2, k=2)

    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
 
    list_p1s = []
    list_p2s = []
    for match in good:
        p1 = keypoints1[match.queryIdx]
        p2 = keypoints2[match.trainIdx]
        list_p1s.append(p1)
        list_p2s.append(p2)

    # Define the number of iterations and the threshold for inliers
    num_iterations = 20
    threshold = 0.00001

    # Initialize the best homography matrix to an identity matrix
    best_homography = np.eye(3, 3, dtype=np.float64)

    # Initialize the inlier max count to zero
    max_inlier_count = 0

    # Repeat N times
    for i in range(num_iterations):
        # Randomly select four feature pairs
        indices = np.random.randint(0, len(list_p1s) , size=4)

        points1 = np.array([list_p1s[i].pt for i in indices])
        points2 = np.array([list_p2s[i].pt for i in indices])

        # Compute the homography matrix
        homography, _ = cv2.findHomography(points1, points2)

        # Compute the inliers by transforming the points in the first image and 
        # computing the sum of squared distances between the points and their 
        # mapped position in the second image
        inliers = 0
        for j in range(len(list_p1s)):
            point1 = np.array([list_p1s[j].pt[0], list_p1s[j].pt[1], 1.0])
            point2 = np.array([list_p2s[j].pt[0], list_p2s[j].pt[1], 1.0])
            mapped_point = np.dot(homography, point1)
            #mapped_point /= mapped_point[2]
            error = point2 - mapped_point
            error = np.sum(error * error)
            
            if error < threshold:
                inliers += 1

        # Update the best homography matrix and the inlier max count if a 
        # larger inlier count is found
        if inliers > max_inlier_count:
            max_inlier_count = inliers
            best_homography = homography
    print(best_homography)
    print(max_inlier_count)

    # Get the size of the output image
    #height, width = image1.shape[:2]
    #height2, width2 = image2.shape[:2]
   
    # Transform the first image to align with the second image
    dst = cv2.warpPerspective(image1, best_homography, (image2.shape[1], image2.shape[0])) #wraped image

    # now paste them together
    #dst[0:image1.shape[0], 0:image1.shape[1]] = image1
 

    # Save the output image to a file or display it using OpenCV's GUI functions
    cv2.imwrite('output_image.jpg', dst)



if __name__ == "__main__":
    #handModel = task1()
    #task2(handModel)
    #task3()
    #task4()
    #task5()
    task6()
    