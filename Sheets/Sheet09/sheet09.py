import numpy as np
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename   
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    print(flow.shape)
    flo_file.close()
    return flow

class OpticalFlow:
    def __init__(self):
        # Parameters for Lucas_Kanade_flow()
        self.EIGEN_THRESHOLD = 0.01 # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade
        self.WINDOW_SIZE = [25, 25] # the number of points taken in the neighborhood of each pixel

        # Parameters for Horn_Schunck_flow()
        self.EPSILON= 0.002 # the stopping criterion for the difference when performing the Horn-Schuck algorithm
        self.MAX_ITERS = 1000 # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
        self.ALPHA = 1.0 # smoothness term

        # Parameter for flow_map_to_bgr()
        self.UNKNOWN_FLOW_THRESH = 1000

        self.prev = None
        self.next = None

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32(np.array([self.prev, self.next]))
        frames /= 255.0

        #calculate image gradient
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
        self.It = frames[1]-frames[0]

        return True

        #***********************************************************************************
    # implement Lucas-Kanade Optical Flow 
    # returns the Optical flow based on the Lucas-Kanade algorithm and visualisation result
    def Lucas_Kanade_flow(self):
        flow_x = np.zeros((self.Ix.shape[0],self.Ix.shape[1]))
        flow_y = np.zeros((self.Ix.shape[0],self.Ix.shape[1]))
        #M
        M = np.zeros((2,2))
        b = np.zeros((2,))
        # Kernel for summation, we have 25 x25 window
        kernel = np.ones((25,25))

        
        #create the entries for the M matric each pixel and b
        xy = self.Ix * self.Iy
        xx = self.Ix * self.Ix
        yy = self.Iy * self.Iy

        yt = self.Iy * self.It
        xt = self.Ix * self.It
        
        xy_sum = cv.filter2D(src= xy, ddepth=-1, kernel=kernel)
        xx_sum = cv.filter2D(src=xx, ddepth=-1, kernel=kernel)
        yy_sum = cv.filter2D(src=yy, ddepth=-1, kernel=kernel)

        xt_sum = cv.filter2D(src=xt, ddepth=-1, kernel=kernel)
        yt_sum = cv.filter2D(src=yt, ddepth=-1, kernel=kernel)

        
        for (x, y), element in np.ndenumerate(flow_x):
            M[0][0] = xx_sum[x,y]
            M[1][0] = xy_sum[x,y]
            M[0][1] = xy_sum[x,y]
            M[1][1] = yy_sum[x,y]

            b[0] = xt_sum[x,y] * -1
            b[1] = yt_sum[x,y] * - 1

            s = np.linalg.solve(M,b)
            flow_x[x,y] = s[0]
            flow_y[x,y] = s[1]

            #print(x)
        flow = np.dstack((flow_x,flow_y))
       
        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    # implement Horn-Schunck Optical Flow 
    # returns the Optical flow based on the Horn-Schunck algorithm and visualisation result
    def Horn_Schunck_flow(self):
        flow = None

        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    #calculate the angular error here
    # return average angular error and per point error map
    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        u_groundtruth_flow = groundtruth_flow[:, :, 0]
        u_estimated_flow = estimated_flow[:, :, 0]
        v_groundtruth_flow = groundtruth_flow[:, :, 1]
        v_estimated_flow = estimated_flow[:, :, 1]
        uu = u_groundtruth_flow * u_estimated_flow
        vv = u_groundtruth_flow * u_estimated_flow
        sum = uu + vv
        sum += 1
        aae = None
        aae_per_point = None
        return aae, aae_per_point

#***********************************************************************************
    #calculate the endpoint error here
    # return average endpoint error and per point error map
    def calculate_endpoint_error(self, estimated_flow, groundtruth_flow):
        aee = None
        aee_per_point = None
        return aee, aee_per_point


#***********************************************************************************
    # function for converting flow map to to BGR image for visualisation
    # return bgr image
    def flow_map_to_bgr(self, flow):
        if flow is None:
            return flow
        h, w = flow.shape[:2]
        #normalize
        flow = flow / np.max(flow)

        x_flow = flow[:, :, 0]
        y_flow = flow[:, :, 1]
        angle= (((np.arctan2(x_flow , y_flow) * 180) / np.pi) / 360) * 127
        angle = np.where(angle > 0.0 , angle, np.abs(angle) + 128)


        x_flow_color = cv.applyColorMap(np.uint8(x_flow * 255), cv.COLOR_GRAY2BGR)
        y_flow_color = cv.applyColorMap(np.uint8(y_flow * 255), cv.COLOR_GRAY2BGR)
        #what should we do with the 3rd color ?
        z_color =  cv.applyColorMap(np.uint8( angle), cv.COLOR_GRAY2BGR)

        bgr_image = np.zeros((h, w, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = y_flow_color[:, :, 0]
        bgr_image[:, :, 1] = x_flow_color[:, :, 1]
        bgr_image[:, :, 2] = z_color[:, :, 2]
        
        return bgr_image


if __name__ == "__main__":


    data_list = [
        'data/frame_0001.png',
        'data/frame_0002.png',
        'data/frame_0007.png',
    ]

    gt_list = [
        './data/frame_0001.flo',
        './data/frame_0002.flo',
        './data/frame_0007.flo',
    ]

    Op = OpticalFlow()
    
    for (i, (frame_filename, gt_filemane)) in enumerate(zip(data_list, gt_list)):
        groundtruth_flow = load_FLO_file(gt_filemane)
        img = cv.cvtColor(cv.imread(frame_filename), cv.COLOR_BGR2GRAY)
        if not Op.next_frame(img):
            continue

        flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        aee_lucas_kanade, aee_lucas_kanade_per_point = Op.calculate_endpoint_error(flow_lucas_kanade, groundtruth_flow)


        #flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        #aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow) 
        #aee_horn_schunk, aee_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow)        
       


        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)

        #fig = plt.figure(figsize=(img.shape))

        # Implement vizualization below  
        # Your functions here
        #print(flow_lucas_kanade_bgr.shape)
        #plt.imshow(Op.flow_map_to_bgr(groundtruth_flow))
        plt.imshow(flow_lucas_kanade_bgr)
        plt.show()

        print("*"*20)

    # Collect and display all the numerical results from all the runs in tabular form (the exact formating is up to your choice)
