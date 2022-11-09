import numpy as np
import typing
import math
from scipy import sparse


class Filter:
    """
    Filter that averages mesh properties with neighbors weighted by their distance
    """
    instance = None

    @staticmethod
    def smoothen(x: np.ndarray) -> np.ndarray:
        """
        Static/global access to apply a averaging filter over the mesh
        :param x: your mesh shaped array
        :return: filtered mesh array
        """
        return Filter.instance(x)

    def __init__(self, rmin, size):
        self.size = size
        Filter.instance = self
        self.weights, self.net_weight = self._prep_filter(rmin)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.weights * (x.flatten('F') / self.net_weight.flatten('F')).T).reshape(x.shape, order="F")

    def _prep_filter(self, filter_radius: float) -> typing.Tuple[np.ndarray, np.ndarray]:
        y_nodes, x_nodes, z_nodes = self.size
        total_nodes = x_nodes * y_nodes * z_nodes
        ceil_r = math.ceil(filter_radius)
        # PREPARE FILTER - something to do with ill-posedness
        iH = list()
        jH = list()
        sH = list()  # weight factors
        # iterates through nodes
        for k1 in range(0, z_nodes):
            for i1 in range(0, x_nodes):
                for j1 in range(0, y_nodes):
                    e1 = k1 * x_nodes * y_nodes + i1 * y_nodes + j1  # 1d index
                    # iterates through neighbors
                    for k2 in range(max(k1 - (ceil_r-1), 0), min(k1 + (ceil_r), z_nodes)):
                        for i2 in range(max(i1 - (ceil_r-1), 0), min(i1 + (ceil_r), x_nodes)):
                            for j2 in range(max(j1 - (ceil_r-1), 0), min(j1 + (ceil_r), y_nodes)):
                                e2 = k2 * x_nodes * y_nodes + i2 * y_nodes + j2
                                iH.append(e1)
                                jH.append(e2)
                                sH.append(max(0, filter_radius - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2)))  # neighbor - distance = weight factor
        sH = np.array(sH); iH = np.array(iH); jH = np.array(jH);
        H = sparse.coo_matrix((sH, (iH, jH)), shape=(total_nodes, total_nodes)).tocsc()  # each row is coordinate, each item in row is the corresponding weight
        Hs = H.sum(1)  # the total weights of a node
        return H, Hs
