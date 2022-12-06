import numpy as np
import matplotlib.pylab as plt

observations = np.load('Sheet06\data\observations.npy')


def get_observation(t):
    return observations[t]

def get_data_at_iteration(n, datalist):
    x_1 = [x[0] for x in datalist[:n+1]]
    y_1 = [x[1] for x in datalist[:n+1]]

    return x_1, y_1 


class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.psi = psi
        self.sigma_p = sigma_p
        self.phi = phi
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None
        self.tau = tau

    def init(self, init_state):
        self.state = init_state
        self.covariance = np.identity(init_state.shape[0])

    def track(self, xt):
        
        # State Prediction.  self.state is the previous state (mu_t_minus_1)
        mu_plus = np.matmul(self.psi, self.state)

        # Covariance Prediction.  self.covariance is the previous covariance (Sigma_t_minus_1)
        sigma_plus = self.sigma_p + np.matmul(np.matmul(self.psi, self.covariance), self.psi.T) 

        # Kalman Gain.  
        K = np.matmul(np.matmul(sigma_plus, self.phi.T), np.linalg.inv(self.sigma_m + np.matmul(np.matmul(self.phi, sigma_plus), self.phi.T)))

        # State Update.  xt is the current observation 
        self.state = mu_plus + np.matmul(K, (xt - np.matmul(self.phi, mu_plus)))   # TODO mu_m ????

        # Covariance Update.
        self.covariance = np.matmul((np.identity(self.state.shape[0]) - np.matmul(K, self.phi)), sigma_plus)  


    def get_current_location(self):
        curr_location = np.matmul(self.phi, self.state)
        return curr_location


def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    return track

def get_world_model():
    psi = np.array([[1, 0, 1.5, 0],
                    [0, 1, 0, 0.5],
                    [0, 0, -1, 0],
                    [0, 0, 0, -2]])
    sp = 0.001
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp, 0],
                        [0, 0, 0, sp]])

    phi = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
    sm = 0.05
    sigma_m = np.array([[sm, 0],
                        [0, sm]])

    return psi, sigma_p, phi, sigma_m 

def main():

    psi, sigma_p, phi, sigma_m = get_world_model()
    
    init_state = np.array([-10, -15, 1, -2])

    kalman_filter = KalmanFilter(psi, sigma_p, phi, sigma_m, 0.1)
    kalman_filter.init(init_state)

    track = perform_tracking(kalman_filter)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    num_steps = len(observations) 
    for n in range(num_steps):
        ax.clear()
        o_x, o_y = get_data_at_iteration(n, observations) 
        t_x, t_y = get_data_at_iteration(n, track) 
         
        ax.plot(o_x, o_y, 'g', label='observations')
        ax.plot(t_x, t_y, 'y', label='Kalman')
        ax.legend()
        plt.pause(0.01)
    plt.pause(3)

if __name__ == "__main__":
    main()
