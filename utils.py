import numpy as np
import matplotlib.pyplot as plt
import math


def plot_spatial_rf(U, n=None, color='black', size=(10,10)):
    # assume U is num_neuron x dim
    fig, ax = plt.subplots(figsize=size)
    ax.cla()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if n is None:
        n = U.shape[0]

    M = int(math.sqrt(n))
    N = M if M ** 2 == n else M + 1
    D = int(math.sqrt(U.shape[1]))

    panel = np.zeros([M * D, N * D])
    # draw
    for i in range(M):
        for j in range(N):
            if i * M + j < n:
                panel[i*D:(i+1)*D,j*D:(j+1)*D] = U[i * M + j].reshape(D, D)

    ax.imshow(panel, "gray")
    plt.setp(ax.spines.values(), color=color)
    return fig, ax
