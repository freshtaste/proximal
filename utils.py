import numpy as np


def bartlett(i,J):
    if np.abs(i) <= J:
        return 1 - np.abs(i)/(J+1)
    else:
        return 0
    
    
def hac(G, J, kernel=bartlett):
    T, K = G.shape
    omega = np.zeros((K,K))
    for j in range(-J, J+1):
        k = bartlett(j,J)
        idx0, idx1 = max(1,j+1)-1, min(T, T+j)-1
        omega += k*(G[idx0:idx1+1].T @ G[idx0-j:idx1+1-j])/T
    return omega
