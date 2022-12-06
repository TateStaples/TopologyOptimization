__author__ = "Tate Staples"

import numpy as np
import typing


class LoadCase:
    """
    Simplified interface for applying load and affixing portions of the structure
    """
    net_importance = 0  # todo: add multiple load case support.

    def __init__(self, shape, importance):
        self.importance = importance
        self.net_importance += importance
        y_nodes, x_nodes, z_nodes = self.shape = shape
        self.num_dofs = (x_nodes + 1) * (y_nodes + 1) * (z_nodes + 1)  # number of degrees of freedom
        self._forces = np.zeros((self.num_dofs, 3))
        self._dof_freedom = np.ones((self.num_dofs, 3), dtype=bool)

    # accessor method
    @property
    def force(self): return self._forces.reshape((self.num_dofs*3, 1))
    @property
    def free_dof(self): return np.where(self._dof_freedom.flatten() > 0)[0]
    @property
    def relative_importance(self): return self.importance / self.net_importance

    def add_force(self, force, mask):
        """
        Applies force at the specified positions
        :param force: the force (fx, fy, fy) to be applied
        :param mask: where to be applied. True where to be applied, False everywhere else
        :return: a reference to the modified self for chained commands
        """
        self._forces[self._get_dof(mask)] = force
        return self

    def affix(self, mask, lock_x=True, lock_y=True, lock_z=True):
        """
        Affix parts of the structure
        :param mask: where should be affixed
        :param lock_x: whether to affix the x dof at the masked indices
        :param lock_y: whether to affix the y dof at the masked indices
        :param lock_z: whether to affix the z dof at the masked indices
        :return: modified reference to self for chained commands
        """
        self._dof_freedom[self._get_dof(mask)] = (not lock_x, not lock_y, not lock_z)
        return self

    def _get_dof(self, mask):
        """
        Private util method for converting mask into dof indices
        :param mask: where in the mesh to references
        :return: indices in terms of the DOF shape organization
        """
        y, x, z = mask.shape
        i2, i1, i3 = np.where(mask > 0)
        load = i3 * x * y + i1 * y + (y - 1 - i2)
        return load

    # preset load cases
    @staticmethod
    def cantilever(shape: typing.Tuple[int, int, int]):  # affiix side, push other side
        y_nodes, x_nodes, z_nodes = shape
        force_location = np.zeros((y_nodes + 1, (x_nodes + 1), (z_nodes + 1)), dtype=bool)
        fix_location = force_location.copy()
        force_location[0, x_nodes, :] = True
        fix_location[:, 0, :] = True
        load_case = LoadCase(shape, 1).add_force((0, -1, 0), force_location).affix(fix_location)
        return load_case

    @staticmethod
    def compress(shape: typing.Tuple[int, int, int]):  # press from top and bottom
        y_nodes, x_nodes, z_nodes = shape
        force_location = np.zeros((y_nodes + 1, (x_nodes + 1), (z_nodes + 1)), dtype=bool)
        fix_location = force_location.copy()
        f2 = force_location.copy()
        force_location[y_nodes, :, :] = True
        f2[0, :, :] = True
        # fix_location[0, :, :] = True
        load_case = LoadCase(shape, 1).add_force((0, -1e4, 0), force_location).affix(fix_location, lock_x=False, lock_z=False).add_force((0, 1e4, 0), f2)
        return load_case

    @staticmethod
    def table(shape: typing.Tuple[int, int, int]):  # affixed bottom, load from top
        y_nodes, x_nodes, z_nodes = shape
        force_location = np.zeros((y_nodes + 1, (x_nodes + 1), (z_nodes + 1)), dtype=bool)
        fix_location = force_location.copy()
        # f2 = force_location.copy()
        force_location[y_nodes, :, :] = True
        # f2[0, :, :] = True
        fix_location[0, :, :] = True
        load_case = LoadCase(shape, 1).add_force((0, -10000, 0), force_location).affix(fix_location, lock_x=False,lock_z=False)  # .add_force((0, 10000, 0), f2)
        return load_case

    @staticmethod
    def torsion(shape, r):  # twist top, affix bottom
        y_nodes, x_nodes, z_nodes = shape
        force_location = np.zeros((y_nodes + 1, (x_nodes + 1), (z_nodes + 1)), dtype=bool)
        fix_location = force_location.copy()
        fix_location[0, :, :] = True
        load_case = LoadCase(shape, .1).affix(fix_location)
        cx, cz = (x_nodes + 1)/2, (z_nodes + 1)/2
        mult = 1e4
        for x in range(x_nodes + 1):
            for z in range(z_nodes + 1):
                dis = np.sqrt((x-cx)**2 + (z-cz)**2)
                if dis<r-1: continue
                m = force_location.copy()
                m[y_nodes, x, z] = True
                f = ((z-cz)*mult, 0, -(x-cx)*mult)
                load_case.add_force(f, m)
        return load_case
