import numpy as np


def passive_cylinder(shape):
    height, diameter, d2 = shape
    assert diameter == d2, "Not square face"
    center = (diameter-1) / 2, (diameter-1)/ 2
    radius = diameter / 2
    Y, X = np.ogrid[:diameter, :diameter]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    x1, z1 = np.where(dist_from_center <= radius)
    mask = np.ones(shape, dtype=bool)
    mask[:, x1, z1] = False
    return mask


def dof_passive(shape):
    height, diameter, d2 = shape
    assert diameter == d2, "Not square face"
    center = diameter / 2, diameter / 2
    radius = diameter / 2
    Y, X = np.ogrid[:diameter+1, :diameter+1]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    x1, z1 = np.where(dist_from_center <= radius)
    mask = np.ones((height+1, d2+1, d2+1), dtype=bool)
    mask[:, x1, z1] = False
    return mask

units_per_meter = 1
def scale_stress(x): return x / units_per_meter**2  # N/m^2 --> N//d^2
def unscale_stress(x): return x * units_per_meter**2  # N/d^2 --> N/m^2
