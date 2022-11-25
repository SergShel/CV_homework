import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))
    return ax


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 25  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

# ------------------------

def calculate_external_energy(image):
    """ calculate external energy
    :param image: gray-scale image
    :return: external energy
    """
    # calculate horizontal and vertical gradients of the image (use cv2.Sobel)
    x_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    y_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    E_ext = - (x_sobel ** 2 + y_sobel ** 2)
    # normalize the energy to [-1, 0]
    E_ext = - E_ext / np.min(E_ext).astype(np.float32)
    E_ext = cv2.GaussianBlur(E_ext, (0, 0), 5)
    # normalize the energy to [-1, 0]
    E_ext = - E_ext / np.min(E_ext)
    return E_ext

def calculate_elasticity_of_pair(node1, node2, dist):
    """
    Calculates elasticity of the pair of nodes.
    :param node1: first node
    :param node2: second node
    :param dist: mean distance between all nodes
    :return: elasticity of the pair
    """
    return (np.linalg.norm(node1 - node2) - dist) ** 2


def get_cost_matrix(E_ext, V, size, alpha=0.1, beta=0.1):
    """
    Returns cost matrix for all relative positions.
    :param E_ext: external energy
    :param V: snake vertices
    :param size: size of the 2d matrix with relative positions
    :return: cost matrix
    """
    relative_positions = np.array([[x, y] for x in range(-(size//2), size//2 + 1) for y in range(-(size//2), size//2 + 1)])
    mean_dist = np.mean([np.linalg.norm(V[i] - V[i-1]) for i in range(1, len(V))])

    num_states = relative_positions.shape[0]
    num_vertices = V.shape[0]



    cost_matrix = np.zeros((num_vertices, num_states), dtype=np.float32)

    backtracking_matrix = np.ones((num_vertices, num_states), dtype=np.int32)
    
    for i, v in enumerate(V):
        for j, r_1 in enumerate(relative_positions):
            curr_pos = v + r_1
            U_ij = E_ext[curr_pos[1], curr_pos[0]] * beta
            #set cost to very small value with respect to Machine limits for floating point types. 
            #This is to avoid division by zero in the next step
            cost_matrix[i, j] = np.finfo(np.float32).max

            for k, r_2 in enumerate(relative_positions):
                prev_v = V[i-1] + r_2
                # calculate elasticity of the pair
                elast_ki = calculate_elasticity_of_pair(curr_pos, prev_v, mean_dist) * alpha
                # calculate cost
                cost = U_ij + elast_ki
                if cost < cost_matrix[i, j]:
                    cost_matrix[i, j] = cost
                    backtracking_matrix[i, j] = k

    return cost_matrix, backtracking_matrix


def backtrack(cost_matrix, backtracking_matrix):
    """
    Backtracks the cost matrix to find the best path.
    :param cost_matrix: cost matrix
    :param backtracking_matrix: backtracking matrix
    :return: best path
    """
    num_vertices = cost_matrix.shape[0]
    num_states = cost_matrix.shape[1]

    # find the best state for the last vertex
    best_state = np.argmin(cost_matrix[num_vertices-1])
    best_path = [best_state]
    for i in range(num_vertices-2, -1, -1):
        best_state = backtracking_matrix[i+1, best_state]
        best_path.append(best_state)
    best_path.reverse()
    return best_path


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 205
    alpha = 0.001
    beta = 1
    size = 3


    # ------------------------
    # your implementation here

    # ------------------------
    # calculate external energy
    E_ext = calculate_external_energy(Im)
    #calculate relative positions
    relative_positions = np.array([[x, y] for x in range(-(size//2), size//2 + 1) for y in range(-(size//2), size//2 + 1)])    

    for t in range(n_steps):
        # ------------------------
        # your implementation here

        # ------------------------
        # calculate cost matrix
        cost_matrix, backtracking_matrix = get_cost_matrix(E_ext, V, size, alpha, beta)
        # backtrack
        trace = backtrack(cost_matrix, backtracking_matrix)
        # update vertices
        new_V = []
        for i, v in zip(trace, V):
            v += relative_positions[i]
            new_V.append(v)
        V = np.array(new_V)

        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.05)

    plt.pause(2)

if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=100)
