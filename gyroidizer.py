import scipy.interpolate
from numpy import pi
from vedo import *


def get_struct_param(density): return (1-density-0.501) / 0.3325


def lerp(density, x, y, z):  # check if this works
    grid = [range(0, i) for i in density.shape]
    return scipy.interpolate.interpn(grid, density, (y, x, z))


def gyroid(x, y, z, t, scale):
    f = 2 * pi * scale
    return cos(f*x)*sin(f*y) + cos(f*y)*sin(f*z) + cos(f*z)*sin(f*x) - t


def optimized_gyroid(x, y, z, t, scale):
    v = gyroid(x, y, z, t, scale)
    f = 2 * scale
    penal = (0.45*t - 0.58)*(cos(f*x)*cos(f*y)+cos(f*y)*cos(f*z)+cos(f*z)*cos(f*x))
    indices = np.where(abs(v)>1.41)
    v[indices] -= penal[indices]
    return v


def gyroidize(density, resolution=15j, scale=1):
    print("beginning gyroidization process")
    y_units, x_units, z_units = density.shape
    x, y, z = np.mgrid[0:x_units-1:(resolution*x_units), 0:y_units-1:(resolution*y_units), 0:z_units-1:(resolution*z_units)]
    volume = optimized_gyroid(x, y, z, get_struct_param(lerp(density, x, y, z)), scale)
    mesh = gen_mesh(volume)
    mesh.write("Structure.stl")
    display(mesh)


def compute_volume(resolution, x_units, y_units, z_units):
    x, y, z = np.mgrid[0:x_units:resolution*x_units, 0:y_units:resolution*y_units,0:z_units:resolution*z_units]
    return optimized_gyroid(x, y, z, strut_param)


def gen_mesh(volume):
    # Create a Volume, take the isosurface at 0, smooth and subdivide it
    surface = Volume(volume).isosurface(0).smooth().lw(1)
    solid = TessellatedBox(n=volume.shape).alpha(1)
    solid.cut_with_mesh(surface)
    gyr = merge(surface, solid)
    return gyr


def display(mesh):
    plotter = Plotter(bg='wheat', bg2='lightblue', axes=5)
    # plotter.add_ambient_occlusion(10)
    print("showing")
    plotter.show(mesh)


if __name__ == '__main__':
    resolution = 15j
    strut_param = get_struct_param(0.5)
    vol = compute_volume(resolution, 4, 4, 4)
    mesh = gen_mesh(vol)
    mesh.write('Gyroid.stl')
    # display(mesh)
