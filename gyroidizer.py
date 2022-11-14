import matplotlib.pyplot as plt
import scipy.interpolate
from numpy import pi
from vedo import *
import numpy as np


def get_struct_param(density): return (1-density-0.501) / 0.3325


def lerp(density, x, y, z):  # check if this works
    grid = [range(0, i) for i in density.shape]
    ep = 1e-5
    x, y, z = np.minimum(x, density.shape[1]-1), np.minimum(y, density.shape[0]-1), np.minimum(z, density.shape[2]-1)
    return scipy.interpolate.interpn(grid, density, (y, x, z))


def gyroid(x, y, z, t, scale):
    f = 2 * pi * scale
    return cos(f*x)*sin(f*y) + cos(f*y)*sin(f*z) + cos(f*z)*sin(f*x) - t


def optimized_gyroid(x, y, z, t, scale):
    v = gyroid(x, y, z, t, scale)
    # f = 2 * pi * scale
    # penal = (0.45*t - 0.58)*(cos(f*x)*cos(f*y)+cos(f*y)*cos(f*z)+cos(f*z)*cos(f*x))
    # indices = np.where(abs(v)>1.41)
    # v[indices] -= penal[indices]
    return v


def gyroidize(density, resolution=15j, scale=1):
    print("beginning gyroidization process")
    y_units, x_units, z_units = density.shape
    x, y, z = np.mgrid[0:x_units-1:(resolution*x_units*scale), 0:y_units-1:(resolution*y_units*scale), 0:z_units-1:(resolution*z_units*scale)]
    volume = optimized_gyroid(x, y, z, get_struct_param(lerp(density, x, y, z)), scale)
    mesh = gen_mesh(volume)
    mesh.write("data/stl/Structure.stl")
    display(mesh)
    return mesh


def compute_volume(resolution, x_units, y_units, z_units):
    x, y, z = np.mgrid[0:x_units:resolution*x_units, 0:y_units:resolution*y_units,0:z_units:resolution*z_units]
    return optimized_gyroid(x, y, z, strut_param, 1)


def gen_mesh(volume):
    # Create a Volume, take the isosurface at 0, smooth and subdivide it
    surface = Volume(volume).isosurface(0).smooth() #.lw(1)
    # solid = TessellatedBox(n=volume.shape).alpha(1) if envelope is None else envelope
    x, y, z = volume.shape
    # surface.cut_with_cylinder((x//2, 0, z//2), axis=(0, 1, 0), r=x//2)
    # gyr = surface.fill_holes()
    gyr = surface
    return gyr


def display(mesh):
    plotter = Plotter()#bg='wheat', bg2='lightblue', axes=5)
    # plotter.add_ambient_occlusion(10)
    print("showing")
    plotter.show(mesh)

def fill_holes(side, c = -1e300):
    return (side < 0) * side + c

def in_cylinder(resolution, x_units, y_units, z_units, scale = 1):
    x, y, z = np.mgrid[0:x_units-1:(resolution*x_units*scale), 0:y_units-1:(resolution*y_units*scale), 0:z_units-1:(resolution*z_units*scale)]
    l = x_units - 1
    w = y_units - 1
    return (x - l/2)**2 + (y - w/2)**2 - (l/2)**2 # * ((l/2)**2 > (x - l/2)**2 + (y - w/2)**2)

if __name__ == '__main__':
    resolution = 16j
    strut_param = get_struct_param(0.2)
    vol = compute_volume(resolution, 2, 2, 2)

    # attempt to cut cylinder without vedo
    cut = in_cylinder(resolution, 2, 2, 2)

    # attempt at filling holes manually
    vol[0, :, :] = fill_holes(vol[0, :, :])
    vol[:, 0, :] = fill_holes(vol[:, 0, :])
    vol[:, :, 0] = fill_holes(vol[:, :, 0])
    x, y, z = vol.shape
    vol[x - 1, :, :] = fill_holes(vol[x - 1, :, :])
    vol[:, y - 1, :] = fill_holes(vol[:, y - 1, :])
    vol[:, :, z - 1] = fill_holes(vol[:, :, z - 1])

    # cap the bottom and top of the cylinder
    cut[0, :, :] = fill_holes(cut[0, :, :])
    cut[:, 0, :] = fill_holes(cut[:, 0, :])
    cut[:, :, 0] = fill_holes(cut[:, :, 0])
    x, y, z = cut.shape
    cut[x - 1, :, :] = fill_holes(cut[x - 1, :, :])
    cut[:, y - 1, :] = fill_holes(cut[:, y - 1, :])
    cut[:, :, z - 1] = fill_holes(cut[:, :, z - 1])

    mesh = gen_mesh(vol)
    cut_mesh = gen_mesh(cut)

    # display(cut_vol)

    # mesh.boolean("plus", cut_mesh)

    print(mesh.is_closed())
    cut_mesh.color("green")
    # mesh.write('C:/Users/skoun/OneDrive - Duke University/Documents/Courses/EGR 101/STL/Gyroid_16.stl')
    
    plotter = Plotter()
    print("showing")
    plotter.show(mesh, cut_mesh)
