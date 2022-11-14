import numpy as np
import scipy.interpolate
from numpy import pi
from vedo import *


def get_struct_param(density): return (1-density-0.501) / 0.3325


def lerp(density, x, y, z):  # check if this works
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
    dx, dy, dz = -np.pi/4, -np.pi/4, -np.pi/4  # phase shift to improve print-ability
    return cos(f*x+dx)*sin(f*y+dy) + cos(f*y+dy)*sin(f*z+dz) + cos(f*z+dz)*sin(f*x+dx) - t


def optimized_gyroid(x, y, z, t, scale):
    v = gyroid(x, y, z, t, scale)
    f = 2 * scale
    penal = (0.45*t - 0.58)*(cos(f*x)*cos(f*y)+cos(f*y)*cos(f*z)+cos(f*z)*cos(f*x))
    indices = np.where(abs(v)>1.41)
    v[indices] -= penal[indices]

    r = x.max()/2
    center_x, center_y = r, r
    fx, cx, fz, cz = np.floor(x*2)/2, np.ceil(x*2/2), np.floor(z*2)/2, np.ceil(z*2)/2
    in_cylinder = (sqrt((fx-r)**2 + (fz-r)**2)<r) * (sqrt((cx-r)**2 + (fz-r)**2)<r) * (sqrt((fx-r)**2 + (cz-r)**2)<r) * (sqrt((cx-r)**2 + (cz-r)**2)<r)
    return v*in_cylinder-1e-8


def gyroidize(density, resolution=15j, scale=1):
    print("beginning gyroidization process")
    y_units, x_units, z_units = density.shape
    x, y, z = np.mgrid[0:x_units-1:(resolution*x_units*scale), 0:y_units-1:(resolution*y_units*scale), 0:z_units-1:(resolution*z_units*scale)]
    volume = optimized_gyroid(x, y, z, get_struct_param(lerp(density, x, y, z)), scale)
    print("Volume Density:", (volume > 0).mean())
    mesh = gen_mesh(volume)
    mesh.write("data/stl/Structure.stl")
    display(mesh)
    return mesh


def compute_volume(resolution, x_units, y_units, z_units):
    x, y, z = np.mgrid[0:x_units:resolution*x_units, 0:y_units:resolution*y_units,0:z_units:resolution*z_units]
    return optimized_gyroid(x, y, z, strut_param, 1)


def gen_mesh(volume):
    x, y, z = volume.shape
    # Create a Volume, take the isosurface at 0, smooth and subdivide it
    v = Volume(volume)
    surface = v.isosurface(0).smooth().lw(1)
    # solid = TessellatedBox(n=(x-1, y-1, z-1)).alpha(1).triangulate()
    # solid.cut_with_mesh(surface)
    # gyr = merge(solid, surface).fill_holes()
    surface.cut_with_cylinder((x//2, 0, z//2), axis=(0, 1, 0), r=x//2)
    gyr = surface
    # gyr = surface.fill_holes()
    return gyr


def display(mesh):
    axes = Axes(mesh)
    plotter = Plotter(axes=4)#bg='wheat', bg2='lightblue', axes=5)
    # plotter.add_ambient_occlusion(10)
    print("showing")
    plotter.show(mesh, axes)


if __name__ == '__main__':
    resolution = 20j
    strut_param = get_struct_param(0.1)
    vol = compute_volume(resolution, 5, 5, 5)
    print((vol>0).mean())
    mesh = gen_mesh(vol)
    # mesh.color("green")
    mesh.write('data/stl/Gyroid.stl')
    display(mesh)
