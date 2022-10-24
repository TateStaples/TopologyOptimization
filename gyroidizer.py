import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi
from skimage import measure
from stl import mesh
# matplotlib.use("TkAgg")
# https://github.com/pbauermeister/matplotlib_gyroid


def get_struct_param(density): return (1-density-0.501) / 0.3325


def lerp(density, x, y, z):  # check if this works
    int_x, rem_x = x.astype(int), x-x.astype(int)
    int_y, rem_y = y.astype(int), y-y.astype(int)
    int_z, rem_z = z.astype(int), z-z.astype(int)
    origin = density[int_y, int_x, int_z]
    return origin
    return np.mean([
        (1-rem_x)*origin+rem_x*density[int_y, int_x+1, int_z],
        (1-rem_y)*origin+rem_y*density[int_y+1, int_x, int_z],
        (1-rem_z)*origin+rem_z*density[int_y, int_x, int_z+1],
    ])


def gyroid(x, y, z):
    return cos(2*pi*x)*sin(2*pi*y) + cos(2*pi*y)*sin(2*pi*z) + cos(2*pi*z)*sin(2*pi*x) - strut_param

def optimized_gyroid(x, y, z, strut_param):
    pass


def gyroidize(density, resolution=25j):
    y_units, x_units, z_units = density.shape
    x, y, z = np.mgrid[0:x_units-1:(resolution*x_units), 0:y_units-1:(resolution*y_units), 0:z_units-1:(resolution*z_units)]

    volume = cos(2*pi*x) * sin(2*pi*y) + cos(2*pi*y)*sin(2*pi*z) + cos(2*pi*z) * sin(2*pi*x) - get_struct_param(lerp(density, x, y, z))
    vertices, faces, normals, vals = measure.marching_cubes(
        volume, level=0,
        spacing=(0.1, 0.1, 0.1),
        allow_degenerate=False)
    return vertices, faces


def compute_volume(resolution, x_units, y_units, z_units):
    x, y, z = np.mgrid[0:x_units:resolution, 0:y_units:resolution,0:z_units:resolution]
    volume = gyroid(x, y, z)
    print((volume>0).mean())
    vertices, faces, normals, vals = measure.marching_cubes(
        volume, 0,
        spacing=(0.1, 0.1, 0.1),
        allow_degenerate=True)
    return vertices, faces


def save_stl(vertices, faces, filename):
    data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
    m = mesh.Mesh(data, remove_empty_areas=False)
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = vertices[f[j], :]
    m.save(filename)#, mode=stl.Mode.ASCII)


def plot(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
        cmap='ocean',
        lw=1)
    plt.ion()
    plt.show()


if __name__ == '__main__':
    resolution = 30j
    strut_param = get_struct_param(0.8)
    vertices, faces = compute_volume(resolution, 2, 2, 2)
    save_stl(vertices, faces, 'Gyroid.stl')
    plot(vertices, faces)
    plt.pause(0.02)
