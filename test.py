import numpy as np


def proj(onto, vec): return np.dot(vec, onto) / np.dot(onto, onto) * onto
if __name__ == '__main__':
    A = np.matrix([
        [1, 1],
        [1, 1]
    ])
    B = np.matrix([
        [1, 2],
        [1, 2]
    ])
    print(A*A)