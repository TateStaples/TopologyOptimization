import numpy as np
import matplotlib
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from scipy.interpolate import interpn


class Display:
    def __init__(self, shape):
        """
        Setup the shared fig and parameters for future calls
        :param shape: the dimensions of the structure array
        """
        matplotlib.use("TkAgg")
        (y_nodes, x_nodes, z_nodes) = shape
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_box_aspect((x_nodes, z_nodes, y_nodes))
        plt.ion()
        self.shapes = list()
        self.max = 0

    def display_3d(self, structure: np.ndarray, strain: np.ndarray):
        """
        Display the structure into Matplotlib
        Opacity = density, strain= blue->red scale
        :param structure: the densities at each location
        :param strain: the strain or stress at each element
        :return:
        """
        self.shapes.append((structure.copy(), strain.copy()))
        structure = self._restructure(structure)
        strain = self._restructure(strain)
        rgba = self._colors(structure, strain)
        # https://www.tutorialspoint.com/how-to-get-an-interactive-plot-of-a-pyplot-when-using-pycharm
        blocks = np.zeros(structure.shape, dtype=bool)
        blocks[structure > 0.05] = True  # hide everything below default density
        self.ax.clear()
        self.ax.voxels(blocks, facecolors=rgba)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _colors(self, structure, strain):
        """Private method to calculate the face colors in the voxel array"""
        y_nodes, x_nodes, z_nodes = shape = structure.shape
        self.max = max(self.max, strain.max())
        strain = np.minimum(1.0, strain / self.max)
        total_nodes = x_nodes * y_nodes * z_nodes
        hue = 2 / 3 - strain * 2 / 3  # get red to blue hue depending on displacement
        saturation = np.ones(shape)  # always high saturation
        value = np.ones(shape)  # always bright
        hsv = np.stack((hue, saturation, value), axis=3)  # build color
        rgb = hsv_to_rgb(hsv.reshape((total_nodes, 3))).reshape((*shape, 3))  # convert to accepted format
        alpha = structure.reshape((*shape, 1))
        rgba = np.concatenate((rgb, alpha), axis=3)  # same thing with tranparency equal to density
        return rgba

    def _restructure(self, structure):
        """Reshape the array from calculation shape the Matplot shape"""
        return np.swapaxes(np.flip(np.swapaxes(structure, 0, 2), 2), 0, 1)

    def _animate(self, frame):
        """Draw the frame of the structure on iteration {frame}"""
        struct, strain = self.shapes[frame]
        self.display_3d(struct, strain)
        return self.ax

    def make_animation(self, fname):
        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(self.fig, self._animate, frames=len(self.shapes), interval=100)
        # Save as gif
        ani.save(f'data/ani/{fname}.gif', fps=5)

    def save(self, filename):
        """Save the current state of the display to the imgs data folder"""
        self.fig.savefig("data/imgs/"+filename)

    def show(self):
        """Show the Display state"""
        self.fig.show()


# saving and load of struct densities
def save(structure: np.ndarray, filename: str) -> None:
    np.save("data/struct/"+filename+".npy", structure)


def load(filename: str) -> np.ndarray:
    return np.load("data/struct/"+filename+".npy")


def load_cache(filename: str, radius: int, height: int):
    """Load the cached structure and morph into the new shape"""
    original = load(filename)
    y, x, z = original.shape
    Y, X, Z = np.mgrid[0:y:y/height, 0:x:x/radius/2, 0:z:z/radius/2]
    return interpn([range(0, i) for i in original.shape], original, (Y, X, Z))