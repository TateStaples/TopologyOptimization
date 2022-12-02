__author__ = "Tate Staples"

import scipy.interpolate
from numpy import pi
from vedo import *


def get_struct_param(density: np.ndarray) -> np.ndarray:
    """
    Converts density into the inequality bound for the gyroid equation
    :param density: float or array of 0-1 values representing density
    :return: float or array of struct values -1.5->1.5 representing inequality thresholds
    """
    return (1-density-0.501) / 0.3325


def lerp(density: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Create an density array of shape (y, x, z) given a density map of a different resolution.
    Interpolates between values in the density map
    :param density: reference source of target densities
    :param x: mgrid of the x coordinate of that part of the array
    :param y: mgrid of the x coordinate of that part of the array
    :param z: mgrid of the x coordinate of that part of the array
    :return: new density map
    """
    grid = [range(0, i) for i in density.shape]
    x, y, z = np.minimum(x, density.shape[1]-1), np.minimum(y, density.shape[0]-1), np.minimum(z, density.shape[2]-1)
    return scipy.interpolate.interpn(grid, density, (y, x, z))


def sheet_gyroid(x, y, z, t, scale):
    """
    Equation for a sheet gyroid. gyroid(x, y, z) with bounds of t.
    Used for some other gyroid project but cant get thin enough for our work
    :param x, y, z: the coordinates of each point in the final grid
    :param t: the struct param for how thick to make the gyroid (get_struct_param doesn't work for this)
    :param scale: how to adjust the frequency/size of the gyroids (2 = 2 gyroids per unit)
    :return voxelized volume
    """
    f = 2 * pi * scale
    sheet = cos(f * x) * cos(f * y) + sin(f * y) * cos(f * z) + sin(f * z) * sin(f * x)
    t1 = sheet < t
    t2 = -t < sheet
    return t1 * t2 - 1


def gyroid(x, y, z, t, scale):
    """
    Calculate the value of each point based off the gyroid equation
    :param x, y, z: the coordinates of each point in the final grid
    :param t: the strucutre parameter to determine density. Region is where equation > t (see get_struct_param)
    :param scale: how to adjust the frequency/size of the gyroids (2 = 2 gyroids per unit)
    :return: Corresponding values. Region is where value ≥ 0
    """
    f = 2 * pi * scale
    dx = dy = dz = -np.pi/4  # phase shift to improve print-ability
    return cos(f*x+dx)*sin(f*y+dy) + cos(f*y+dy)*sin(f*z+dz) + cos(f*z+dz)*sin(f*x+dx) - t


def optimized_gyroid(x, y, z, t, scale):
    """
    Modified gyroid equation to prevent low densities from having gaps
    :param x, y, z, t, scale: See gyroid function above
    :return: values of gyroid equation
    """
    v = gyroid(x, y, z, t, scale)
    f = 2 * scale  # todo: this what the paper says but it seems like it should be 2π
    penal = (0.45*t - 0.58)*(cos(f*x)*cos(f*y)+cos(f*y)*cos(f*z)+cos(f*z)*cos(f*x))
    indices = np.where(abs(v)>1.41)
    v[indices] -= penal[indices]

    return v


def project_structure(x, y, z, t, scale):
    """
    Generates full structure. Has the gyroid center, cuts the overhanging parts to ensure printablility, and adds pads
    :param x, y, z, t, scale: See gyroid function above
    :return: values of gyroid equation
    """
    return pad_cylinder(optimized_gyroid(x, y, z, t, scale) * overhang_mask(x, z, scale) - 1e-8)


def overhang_mask(x, z, scale):
    """
    Generates a maks so that only complete gyroid are included in the final strucutre. Useful for printing.
    Gyroids connect together every half phase so crop all points where the corresponding phase is not entirely in cylinder
    :param x, z: coordinates
    :param scale: gyroids per unit distance
    :return:
    """

    scale *= 2  # connections at half scale
    r = x.max() / 2 * scale
    x, z = x[:, 0, :], z[:, 0, :]
    epsilon = 0.00001
    fx, cx, fz, cz = np.floor(x * scale - epsilon), np.ceil(x * scale + epsilon), np.floor(z * scale - epsilon), np.ceil(z * scale + epsilon)
    m1, m2, m3, m4 = sqrt((fx - r) ** 2 + (fz - r) ** 2) < r, sqrt((cx - r) ** 2 + (fz - r) ** 2) < r, sqrt((fx - r) ** 2 + (cz - r) ** 2) < r, sqrt((cx - r) ** 2 + (cz - r) ** 2) < r + epsilon
    in_cylinder = m1 * m2 * m3 * m4
    in_cylinder *= in_cylinder[:, ::-1]  # idk but this makes it work
    return in_cylinder[:, None, :]


def circle_mask(r):
    """
    Create a make to remove material outside the cylinder
    :param r: radius of our circle
    :return: 2r x 2r true/false mask
    """
    center = r
    radius = r-1
    Y, X = np.ogrid[:2*r, :2*r]
    dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

    mask = dist_from_center <= radius
    return mask


def pad_cylinder(v):
    """
    Add some pads for testing to the top and bottom of the structure. Also ensures all edges are capped
    :param v: volume to pad
    :return: padded volume
    """
    pad_shape = v.shape[0], v.shape[1]//8, v.shape[2]
    pad = np.ones(pad_shape) * circle_mask(v.shape[0] / 2)[:, None, :] - 1e-9  # make the pad
    v = np.concatenate((pad, np.concatenate((v, pad), axis=1)), axis=1)  # add the pad to both ends
    # caps the ends
    v[:, (0, v.shape[1] - 1), :] = -1
    v[(0, v.shape[0] - 1), :, :] = -1
    v[:, :, (0, v.shape[2] - 1)] = -1
    return v


def gyroidize(density: np.ndarray, resolution: complex = 15j, scale: float = 1.0) -> Mesh:
    """
    Take a structure from the optimization alg and turn it into stl then display
    :param density: Density map generated by optimization. 0-1 values
    :param resolution: How many voxel samples to take per gyroid. Complex (#j) mean freq
    :param scale: how many gyroids to have per density entry. 2 = twice the gyroids
    :return: the generated mesh
    """
    print("beginning gyroidization process")
    y_units, x_units, z_units = density.shape
    x, y, z = np.mgrid[0:x_units:(resolution*x_units*scale), 0:y_units:(resolution*y_units*scale), 0:z_units:(resolution*z_units*scale)]
    volume = project_structure(x, y, z, get_struct_param(lerp(density, x, y, z)), scale)
    print("Volume Density:", (volume > 0).mean())
    mesh = generate_mesh(volume)
    mesh.write("data/stl/Structure.stl")
    display(mesh)
    return mesh


def gyroid_example(resolution, x_units, y_units, z_units):
    x, y, z = np.mgrid[0:x_units:resolution*x_units, 0:y_units:resolution*y_units,0:z_units:resolution*z_units]
    return optimized_gyroid(x, y, z, strut_param, 1)


def generate_mesh(volume: np.ndarray) -> Mesh:
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


def display(mesh: Mesh):
    """
    Display the mesh in a new window
    :param mesh: object to display
    """
    mesh.color("blue")
    axes = Axes(mesh, xminor_ticks=3)
    plotter = Plotter(axes=4)
    print("showing")
    plotter.show(mesh, axes)


if __name__ == '__main__':
    resolution = 20j
    strut_param = get_struct_param(0.1)
    vol = gyroid_example(resolution, 1, 1, 1)
    mesh = generate_mesh(vol)
    mesh.write('data/stl/Gyroid.stl')
