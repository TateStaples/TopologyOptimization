from vedo import *


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


def gyroid(x, y, z, t):
    pi = 3.14159265
    return cos(2*pi*x)*sin(2*pi*y) + cos(2*pi*y)*sin(2*pi*z) + cos(2*pi*z)*sin(2*pi*x) - t


def optimized_gyroid(x, y, z, t):
    v = gyroid(x, y, z, t)
    penal = (0.45*t - 0.58)*(cos(2*x)*cos(2*y)+cos(2*y)*cos(2*z)+cos(2*z)*cos(2*x))
    indices = np.where(abs(v)>1.41)
    v[indices] -= penal[indices]
    return v


def gyroidize(density, resolution=15j):
    y_units, x_units, z_units = density.shape
    x, y, z = np.mgrid[0:x_units-1:(resolution*x_units), 0:y_units-1:(resolution*y_units), 0:z_units-1:(resolution*z_units)]
    volume = optimized_gyroid(x, y, z, get_struct_param(lerp(density, x, y, z)))
    mesh = gen_mesh(volume)
    mesh.write("Structure.stl")
    mesh.show()


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
    plotter.add_ambient_occlusion(10)
    plotter.show(mesh)


if __name__ == '__main__':
    resolution = 15j
    strut_param = get_struct_param(0.5)
    vol = compute_volume(resolution, 6, 6, 6)
    mesh = gen_mesh(vol)
    mesh.write('Gyroid.stl')
    display(mesh)
