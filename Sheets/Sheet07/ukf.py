import numpy as np
import matplotlib.pylab as plt

observations = np.load('data/observations.npy')


def get_observation(t):
    return observations[t]


class UnscentedKalmanFilter(object):
    def __init__(self, sigma_p, sigma_m):
   
        self.sigma_p = sigma_p
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None
        self.a_0 = 0
        self.a_j = 0
        self.D_w = 0
        self.w_hat = None

        self.count = 0
        self.state_smoothed = []

    def init(self, init_state):
        self.state = init_state
        self.w_hat = init_state
        self.covariance = np.identity(init_state.shape[0]) * 10

        self.a_0 = np.random.uniform(0, 1)
        self.D_w = init_state.shape[0]
        self.a_j = ((1 - self.a_0) / (2 * self.D_w))

    def small_sigma(self, x):
        if x == 0:
            return 1
        return 0

    def f(self, x, e=np.array([0, 0])):
        return np.array([x[0], x[0]*np.sin(x[0])]) + e


    def track(self, xt):
        self.w_hat = self.state.astype(np.float64) 
        w_plus = self.f(self.w_hat)
        mu_plus = np.zeros_like(self.w_hat)
        Sigma_plus = np.zeros_like(self.covariance)


        for j in range(self.D_w * 2 + 1):
            mu_plus += self.a_j * w_plus
            Sigma_plus += self.a_j * (w_plus - mu_plus) @ (w_plus - mu_plus).T + self.sigma_p

        mu_x = np.zeros_like(self.w_hat)
        Sigma_x = np.zeros_like(self.covariance)
        for j in range(self.D_w * 2 + 1):
            mu_x += self.a_j * self.f(self.w_hat)
            Sigma_x += self.a_j * (self.f(self.w_hat) - mu_x) @ (self.f(self.w_hat) - mu_x).T + self.sigma_m

        # Kalman gain now computed from particle
        K = None
        for j in range(self.D_w * 2 + 1):
            
            if K is None:
                K = self.a_j * (np.array([self.w_hat - mu_plus])) @ (np.array([xt - mu_x])).T * np.linalg.inv(Sigma_x)
            else:
                K += self.a_j * (np.array([self.w_hat - mu_plus])) @ (np.array([xt - mu_x])).T * np.linalg.inv(Sigma_x)

        # Measurement update
        self.state = mu_plus + K @ (xt - mu_x)
        self.covariance = Sigma_plus - K @ Sigma_x @ K.T


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

    tracker = UnscentedKalmanFilter(sigma_p, sigma_m)
    tracker.init(init_state)

    track = perform_tracking(tracker)
    print(track)

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
