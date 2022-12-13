import numpy as np
import matplotlib.pylab as plt

observations = np.load('data/observations.npy')


def get_observation(t):
    return observations[t]

# I assume that "The data are the same as page 61-62 in the slide" means
# that we have to use the f-, g- functions and so on 
# from the toy example from the lecture
class ExtendedKalmanFilter(object):
    def __init__(self, sigma_p, sigma_m):
   
        self.sigma_p = sigma_p
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None
        self.psi = None
        self.Y_p = np.identity(2)
        self.phi = np.identity(2)
        self.Y_m = np.identity(2)

        self.count = 0
        self.state_smoothed = []

    def f(self, x, e=np.array([0, 0])):
        return np.array([x[0]+e[0], x[0]*np.sin(x[0])+e[1]])

    def g(self, x, e=np.array([0, 0])):
        return x + e

    def set_psi(self):
        assert self.state is not None
        mu1_t_ = self.state[0]
        self.psi = np.array([[1,                                     0],
                             [np.sin(mu1_t_)+ mu1_t_*np.cos(mu1_t_), 1]])

    def init(self, init_state):
        self.state = init_state
        self.set_psi()
        self.covariance = np.identity(init_state.shape[0]) * 10

    def track(self, xt):
        # State Prediction.  self.state is the previous state (mu_t_minus_1)
        mu_plus = self.f(self.state)

        # Covariance Prediction.  self.covariance is the previous covariance (Sigma_t_minus_1)
        Sigma_plus = self.psi @ self.covariance @ self.psi.T + self.Y_p @ self.sigma_p @ self.Y_p.T 

        # Let Q be equal to 3
        Q = 3
        for q in range(Q):
            mu = mu_plus
            if(q > 0):
                mu = self.state
            
            # Kahlman Gain
            K = Sigma_plus @ self.phi.T @ np.linalg.inv(self.Y_m @ self.sigma_m @ self.Y_m.T + 
                                                        self.phi @ Sigma_plus @ self.phi.T)
            # State Update
            self.state = mu + K @ (xt - self.g(mu))
            self.set_psi()
            # Covariance Update
            self.covariance = (np.identity(2) - K @ self.phi) @ Sigma_plus


    def get_current_location(self):
        return self.state

def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    return track


def get_observations_up_to(n, datalist):
    x_1 = [x[0] for x in datalist[:n+1]]
    y_1 = [x[1] for x in datalist[:n+1]]

    return x_1, y_1 

def main():
    init_state = np.array([0, 0])

    sp = 0.01
    sigma_p = np.array([[sp, 0],
                        [0, sp],])

    sm = 0.001
    sigma_m = np.array([[sm, 0],
                        [0, sm]])

    tracker = ExtendedKalmanFilter(sigma_p, sigma_m)
    tracker.init(init_state)

    track = perform_tracking(tracker)

    # find difference between track and observations
    diff = np.array(track) - np.array(observations)
    print("Mean squared error: ", np.mean(diff**2))
    # Mean squared error = 2.6659573577741185e-11 => 2 lines overlap each other that's why it seams like  one line
    

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    num_steps = len(observations) 
    for n in range(num_steps):
        ax.clear()
        o_x, o_y = get_observations_up_to(n, observations) 
        t_x, t_y = get_observations_up_to(n, track) 

        
        
        ax.plot(o_x, o_y, 'g', label='observations')
        ax.plot(t_x, t_y, 'y', label='EKF')

        
        ax.legend()
        plt.pause(0.1)
    plt.pause(3)



if __name__ == "__main__":
    main()
