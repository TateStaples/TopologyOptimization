__author__ = "Tate Staples"

import math
import numpy as np
from scipy import sparse

filter_weights = filter_net_weights = None


def prep_filter(shape, filter_radius: float):
    """
    Prepare the smoothing constants given a shape and smoothing radius
    :param shape: shape of the structure
    :param filter_radius: range at which to smooth
    :return: None - update the filter_weights & filter_net_weights
    """
    global filter_weights, filter_net_weights
    y_nodes, x_nodes, z_nodes = shape
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
                for k2 in range(max(k1 - (ceil_r - 1), 0), min(k1 + (ceil_r), z_nodes)):
                    for i2 in range(max(i1 - (ceil_r - 1), 0), min(i1 + (ceil_r), x_nodes)):
                        for j2 in range(max(j1 - (ceil_r - 1), 0), min(j1 + (ceil_r), y_nodes)):
                            e2 = k2 * x_nodes * y_nodes + i2 * y_nodes + j2
                            iH.append(e1)
                            jH.append(e2)
                            sH.append(max(0, filter_radius - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2 + (
                                        k1 - k2) ** 2)))  # neighbor - distance = weight factor
    sH = np.array(sH);
    iH = np.array(iH);
    jH = np.array(jH);
    filter_weights = sparse.coo_matrix((sH, (iH, jH)), shape=(total_nodes, total_nodes)).tocsc()  # each row is coordinate, each item in row is the corresponding weight
    filter_net_weights = filter_weights.sum(1)  # the total weights of a node


def filter_sensitivities(x: np.ndarray) -> np.ndarray:
    """
    Gradient of the filtering
    :param x: the sensitivities in shape of the structure
    :return: filtered sensitivities
    """
    return np.array(filter_weights * (x.flatten('F') / filter_net_weights.flatten('F')).T).reshape(x.shape, order="F")


def filter_structure(x: np.ndarray) -> np.ndarray:
    """
    Filtering of density to prevent checkerboarding and singularities
    :param x: the densities in shape of the structure
    :return: filtered densities
    """
    return np.array(filter_weights * x.flatten("F") / filter_net_weights.flatten("F")).reshape(x.shape, order="F")

