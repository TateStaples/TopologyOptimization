__author__ = "Tate Staples"

import numpy as np
import scipy.interpolate
from numpy import pi
from vedo import *


def get_struct_param(density): return (1-density-0.501) / 0.3325


def lerp(density, x, y, z):  # check if this works
    # max_x = x.max()
    # print((x/max_x).shape)
    # return x/max_x+.02
    grid = [range(0, i) for i in density.shape]
    ep = 1e-5
    x, y, z = np.minimum(x, density.shape[1]-1), np.minimum(y, density.shape[0]-1), np.minimum(z, density.shape[2]-1)
    return scipy.interpolate.interpn(grid, density, (y, x, z))


def sheet_gyroid(x, y, z, t, scale):
    f = 2 * pi * scale
    sheet = cos(f * x) * cos(f * y) + sin(f * y) * cos(f * z) + sin(f * z) * sin(f * x)
    t1 = sheet < t
    t2 = -t < sheet
    return t1 * t2 - 1


def gyroid(x, y, z, t, scale):
    f = 2 * pi * scale
    dx = dy = dz = -np.pi/4  # phase shift to improve print-ability
    return cos(f*x+dx)*sin(f*y+dy) + cos(f*y+dy)*sin(f*z+dz) + cos(f*z+dz)*sin(f*x+dx) - t


def optimized_gyroid(x, y, z, t, scale):
    v = gyroid(x, y, z, t, scale)
    f = 2 * scale
    penal = (0.45*t - 0.58)*(cos(f*x)*cos(f*y)+cos(f*y)*cos(f*z)+cos(f*z)*cos(f*x))
    indices = np.where(abs(v)>1.41)
    v[indices] -= penal[indices]

    # v = pad_cylinder(v)
    scale *= 2  # connections at half scale
    r = x.max()/2*scale
    x, z = x[:, 0, :], z[:, 0, :]
    epsilon = 0.00001
    print(scale, x.max()/2*scale)
    fx, cx, fz, cz = np.floor(x*scale-epsilon), np.ceil(x*scale+epsilon), np.floor(z*scale-epsilon), np.ceil(z*scale+epsilon)
    m1, m2, m3, m4 = sqrt((fx-r)**2 + (fz-r)**2)<r, sqrt((cx-r)**2 + (fz-r)**2)<r, sqrt((fx-r)**2 + (cz-r)**2)<r, sqrt((cx-r)**2 + (cz-r)**2)<r+epsilon
    in_cylinder = m1*m2*m3*m4
    in_cylinder *= in_cylinder[:, ::-1]
    t1, t2, t3, t4, t = [np.array([[m[i,j]
                                    for i in range(0, int(x.shape[0]), x.shape[0]//int(round(1/scale*2)+1))]
                                   for j in range(0, int(x.shape[0]), x.shape[0]//int(round(1/scale*2))+1)])
                                    for m in (m1, m2, m3, m4, in_cylinder)]
    return pad_cylinder(v*in_cylinder[:, None, :]-1e-8)


def cylinder_mask(r):
    center = r
    radius = r-1
    print(r, radius)
    Y, X = np.ogrid[:2*r, :2*r]
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

    mask = dist_from_center <= radius
    return mask


def pad_cylinder(v):
    pad_shape = v.shape[0], v.shape[1]//8, v.shape[2]
    pad = np.ones(pad_shape) * cylinder_mask(v.shape[0]/2)[:, None, :] - 1e-9
    v = np.concatenate((pad, np.concatenate((v, pad), axis=1)), axis=1)
    v[:, (0, v.shape[1] - 1), :] = -1
    v[(0, v.shape[0] - 1), :, :] = -1
    v[:, :, (0, v.shape[2] - 1)] = -1
    return v


def gyroidize(density, resolution=15j, scale=1):
    print("beginning gyroidization process")
    y_units, x_units, z_units = density.shape
    x, y, z = np.mgrid[0:x_units:(resolution*x_units*scale), 0:y_units:(resolution*y_units*scale), 0:z_units:(resolution*z_units*scale)]
    volume = optimized_gyroid(x, y, z, get_struct_param(lerp(density, x, y, z)), scale)
    print("Volume Density:", (volume > 0).mean())
    mesh = gen_mesh(volume)
    mesh.write("data/stl/Structure.stl")
    display(mesh)
    return mesh


def compute_volume(resolution, x_units, y_units, z_units):
    x, y, z = np.mgrid[0:x_units:resolution*x_units, 0:y_units:resolution*y_units,0:z_units:resolution*z_units]
    return optimized_gyroid(x, y, z, strut_param, 1)


def gen_mesh(volume) -> Mesh:
    """
    Convert voxel array into a smooth mesh
    :param volume: 3d array of true/false values representing location of material
    :return: Computed mesh
    """
    x, y, z = volume.shape
    # Create a Volume, take the isosurface at 0, smooth and subdivide it
    v = Volume(volume)
    surface = v.isosurface(0).smooth().lw(1)
    # solid = TessellatedBox(n=(x-1, y-1, z-1)).alpha(1).triangulate()
    # solid.cut_with_mesh(surface)
    surface.cut_with_cylinder((x/2, 0, z/2), axis=(0, 1, 0), r=x/2)
    gyr = surface
    # gyr = surface.fill_holes()
    return gyr


def display(mesh):
    """
    Display the mesh in a new window
    :param mesh: object to display
    """
    mesh.color("blue")
    axes = Axes(mesh, xminor_ticks=3)
    plotter = Plotter(axes=4)

    # plotter.add_ambient_occlusion(10)
    print("showing")
    plotter.show(mesh, axes)


if __name__ == '__main__':
    resolution = 20j
    strut_param = get_struct_param(0.1)
    vol = compute_volume(resolution, 1, 1, 1)
    print((vol>0).mean())
    mesh = gen_mesh(vol)
    print(mesh.is_closed())
    # mesh.color("green")
    mesh.write('data/stl/Gyroid.stl')
    display(mesh)
