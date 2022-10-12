import numpy as np
import math
from scipy.sparse import *
import typing
import matplotlib.pyplot as plt
from scipy.sparse import linalg
from matplotlib.colors import hsv_to_rgb
import time


class Material:
    """
    Class the wraps the material properties you are using.
    Used to calculate the stiffness given the density
    """
    # todo: gyroid material - penal and poisson

    def __init__(self, poisson: float = 0.3, max_stiff: float = 1.0, min_stiff: float = 1e-19, penal: float = 3):
        """
        Initialize the properties of for element
        :param poisson: the deformation shrinkage
        :param max_stiff: Young's modulus of solid material (stiffness)
        :param min_stiff: Young's modulus of void-like material
        """
        assert poisson > 0 and max_stiff <= 1 and min_stiff >= 0
        self.poisson = poisson
        self.max_stiff = max_stiff
        self.min_stiff = min_stiff
        self.element_stiffness = self._element_stiffness()
        self.strain_matrix = self._strain_matrix()
        self.elastic_matrix = self._elastic_matrix()
        self.penal = penal

    def _element_stiffness(self): # 3d
        """
        Converts the material properties into a matrix representing the stiffness of a hexhedral (cube) element
        :return: 24x24 array representing the x,y,z stiffness of each corner of the element
        """
        A = np.array([  # todo: figure out where this is derived from
            [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
            [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]
        ])
        k = (1 / 144 * A.T * [1, self.poisson]).sum(1)

        # 24 dof because cube has 8 nodes that can translate 3d each
        def genK(indices):
            return np.array([k[i - 1] for i in indices]).reshape((6, 6))

        # this comes from some wacky integration
        K1 = genK([1, 2, 2, 3, 5, 5,
                   2, 1, 2, 4, 6, 7,
                   2, 2, 1, 4, 7, 6,
                   3, 4, 4, 1, 8, 8,
                   5, 6, 7, 8, 1, 2,
                   5, 7, 6, 8, 2, 1])
        K2 = genK([9, 8, 12, 6, 4, 7,
                   8, 9, 12, 5, 3, 5,
                   10, 10, 13, 7, 4, 6,
                   6, 5, 11, 9, 2, 10,
                   4, 3, 5, 2, 9, 12,
                   11, 4, 6, 12, 10, 13])
        K3 = genK([6, 7, 4, 9, 12, 8,
                   7, 6, 4, 10, 13, 10,
                   5, 5, 3, 8, 12, 9,
                   9, 10, 2, 6, 11, 5,
                   12, 13, 10, 11, 6, 4,
                   2, 12, 9, 4, 5, 3])
        K4 = genK([14, 11, 11, 13, 10, 10,
                   11, 14, 11, 12, 9, 8,
                   11, 11, 14, 12, 8, 9,
                   13, 12, 12, 14, 7, 7,
                   10, 9, 8, 7, 14, 11,
                   10, 8, 9, 7, 11, 14])
        K5 = genK([1, 2, 8, 3, 5, 4,
                   2, 1, 8, 4, 6, 11,
                   8, 8, 1, 5, 11, 6,
                   3, 4, 5, 1, 8, 2,
                   5, 6, 11, 8, 1, 8,
                   4, 11, 6, 2, 8, 1])
        K6 = genK([14, 11, 7, 13, 10, 12,
                   11, 14, 7, 12, 9, 2,
                   7, 7, 14, 10, 2, 9,
                   13, 12, 10, 14, 7, 11,
                   10, 9, 2, 7, 14, 7,
                   12, 2, 9, 11, 7, 14])

        KE = 1 / ((self.poisson + 1) * (1 - 2 * self.poisson)) * np.array([
            [K1, K2, K3, K4],
            [K2.T, K5, K6, K3.T],
            [K3.T, K6, K5.T, K2.T],
            [K4, K3, K2, K1.T]
        ])
        return KE.swapaxes(1, 2).reshape(24, 24)

    def _strain_matrix(self):
        B_1 = np.array([-0.044658, 0, 0, 0.044658, 0, 0, 0.16667, 0,
               0, -0.044658, 0, 0, -0.16667, 0, 0, 0.16667,
               0, 0, -0.044658, 0, 0, -0.16667, 0, 0,
               - 0.044658, -0.044658, 0, -0.16667, 0.044658, 0, 0.16667, 0.16667,
               0, -0.044658, -0.044658, 0, -0.16667, -0.16667, 0, -0.62201,
               - 0.044658, 0, -0.044658, -0.16667, 0, 0.044658, -0.62201, 0]).reshape((6, 8))
        B_2 = np.array([0, -0.16667, 0, 0, -0.16667, 0, 0, 0.16667,
               0, 0, 0.044658, 0, 0, -0.16667, 0, 0,
               - 0.62201, 0, 0, -0.16667, 0, 0, 0.044658, 0,
               0, 0.044658, -0.16667, 0, -0.16667, -0.16667, 0, -0.62201,
               0.16667, 0, -0.16667, 0.044658, 0, 0.044658, -0.16667, 0,
               0.16667, -0.16667, 0, -0.16667, 0.044658, 0, -0.16667, 0.16667]).reshape((6, 8))
        B_3 = np.array([0, 0, 0.62201, 0, 0, -0.62201, 0, 0,
               - 0.62201, 0, 0, 0.62201, 0, 0, 0.16667, 0,
               0, 0.16667, 0, 0, 0.62201, 0, 0, 0.16667,
               0.16667, 0, 0.62201, 0.62201, 0, 0.16667, -0.62201, 0,
               0.16667, -0.62201, 0, 0.62201, 0.62201, 0, 0.16667, 0.16667,
               0, 0.16667, 0.62201, 0, 0.62201, 0.16667, 0, -0.62201]).reshape((6, 8))

        B = np.hstack((B_1, B_2, B_3))
        return B

    def _elastic_matrix(self):
        nu = self.poisson
        D = [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0,0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ]
        return 1 / ((1 + nu) * (1 - 2 * nu)) * np.array(D)

    def stiffness(self, density):
        """
        Calculate stiffness of the material given a material
        :param density: the density of the material
        :return: stiffness matrix
        """
        return self.min_stiff + density ** self.penal * (self.max_stiff - self.min_stiff)

    def gradient(self, density):
        """
        Get the gradient of stiffness by density
        :param density: the density matrix
        :return: the gradient of the stiffness compared to density
        """
        return self.penal * (self.max_stiff - self.min_stiff) * density ** (self.penal - 1)  # see equation 26 of matlab paper


class Filter:
    """
    Filter that averages mesh properties with neighbors weighted by their distance
    """
    instance = None

    @staticmethod
    def smoothen(x: np.ndarray) -> np.ndarray:
        """
        Static/global access to apply a averaging filter over the mesh
        :param x: your mesh shaped array
        :return: filtered mesh array
        """
        return Filter.instance(x)

    def __init__(self, rmin, size):
        self.size = size
        Filter.instance = self
        self.weights, self.net_weight = self._prep_filter(rmin)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.weights * (x.flatten('F') / self.net_weight.flatten('F')).T).reshape(x.shape, order="F")

    def _prep_filter(self, filter_radius: float) -> typing.Tuple[np.ndarray, np.ndarray]:
        y_nodes, x_nodes, z_nodes = self.size
        total_nodes = x_nodes * y_nodes * z_nodes
        ceil_r = math.ceil(filter_radius)
        # PREPARE FILTER - something to do with ill-posedness
        iH = list()
        jH = list()
        sH = list()  # weight factors
        # iterates through nodes
        for k1 in range(0, z_nodes):
            for i1 in range(0, x_nodes):
                for j1 in range(0, y_nodes):
                    e1 = k1 * x_nodes * y_nodes + i1 * y_nodes + j1  # 1d index
                    # iterates through neighbors
                    for k2 in range(max(k1 - (ceil_r-1), 0), min(k1 + (ceil_r), z_nodes)):
                        for i2 in range(max(i1 - (ceil_r-1), 0), min(i1 + (ceil_r), x_nodes)):
                            for j2 in range(max(j1 - (ceil_r-1), 0), min(j1 + (ceil_r), y_nodes)):
                                e2 = k2 * x_nodes * y_nodes + i2 * y_nodes + j2
                                iH.append(e1)
                                jH.append(e2)
                                sH.append(max(0, filter_radius - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2)))  # neighbor - distance = weight factor
        sH = np.array(sH); iH = np.array(iH); jH = np.array(jH);
        H = coo_matrix((sH, (iH, jH)), shape=(total_nodes, total_nodes)).tocsc()  # each row is coordinate, each item in row is the corresponding weight
        Hs = H.sum(1)  # the total weights of a node
        return H, Hs


class FEA:
    """
    Class the performs the analysis of how the structure deforms
    """
    def __init__(self, shape: typing.Tuple[int, int, int], material: Material):
        """
        Initialize the setup for the FEA analysis
        :param shape: The shape of the grid of nodes
        :param material: The material that the model is made of
        """
        self.material = material
        y_nodes, x_nodes, z_nodes = shape
        self.total_nodes = x_nodes * y_nodes * z_nodes
        self.num_dofs = 3 * (x_nodes + 1) * (y_nodes + 1) * (z_nodes + 1)  # number of degrees of freedom - format is [Fx1, Fy1, Fz1, Fx2 ... Fz#]

        # loads and supports
        force_location = np.zeros((y_nodes + 1, (x_nodes + 1), (z_nodes + 1)), dtype=bool)
        fix_location = force_location.copy()
        force_location[y_nodes, :, :] = True
        fix_location[0, :, :] = True
        load_case = LoadCase(shape, 1).add_force((0, -1, 0), force_location).affix(fix_location)
        self.forces = csr_matrix(load_case.force)
        self.free_dofs = load_case.free_dof

        # setup node connections
        node_grd = np.array(range(0, (y_nodes + 1) * (x_nodes + 1))).reshape((y_nodes + 1, x_nodes + 1), order='F')
        node_ids = node_grd[:-1, :-1].reshape((y_nodes * x_nodes, 1), order="F")
        nodeidz = np.array(range(0, (z_nodes - 1) * (y_nodes + 1) * (x_nodes + 1) + 1, (y_nodes + 1) * (x_nodes + 1)))
        node_ids = np.tile(node_ids, (1, len(nodeidz))) + nodeidz
        element_dof_vec = 3 * (node_ids + 1).T.flatten()  # the first node of each cube
        relative_element = np.array([0, 1, 2, *
                (np.array([3, 4, 5, 0, 1, 2]) + 3 * y_nodes), -3, -2, -1,
                *(3 * (y_nodes + 1) * (x_nodes + 1) +
                np.array([0, 1, 2, *(np.array([3, 4, 5, 0, 1, 2]) + 3 * y_nodes), -3, -2, -1]))])
        # 3d
        self.element_dof_mat = np.tile(element_dof_vec, (24, 1)).T + relative_element  # indices of the dof for each element
        self.iK = np.kron(self.element_dof_mat, np.ones((24, 1))).flatten()
        self.jK = np.kron(self.element_dof_mat, np.ones((1, 24))).flatten()

        # physical properties
        self.displacement = np.zeros((self.num_dofs, 1))
        self.global_stiffness = csc_matrix((self.num_dofs, self.num_dofs))
        self.compliance = 0
        self.strain = np.zeros(shape)

        self.total_stress = 0
        self.von_mises_stress = np.zeros((self.total_nodes, 1))  # von Mises stress vector
        self.stress = np.zeros((self.total_nodes, 6))  # stress has 6 components for each

    def displace(self, density: np.ndarray):
        elementwise_stiffness = ((self.material.element_stiffness.flatten()[np.newaxis]).T*(self.material.min_stiff+density.flatten('F')**self.material.penal*(self.material.max_stiff-self.material.min_stiff))).flatten(order='F')  # figure 15 fixme
        global_stiffness = coo_matrix((elementwise_stiffness, (self.iK, self.jK)), shape=(self.num_dofs, self.num_dofs)).tocsc()
        global_stiffness = (global_stiffness + global_stiffness.T) / 2.0  # average with previous?
        # Remove constrained dofs from matrix
        gs = global_stiffness[self.free_dofs, :][:, self.free_dofs]
        fs = self.forces[self.free_dofs, :]
        self.displacement[self.free_dofs, 0] = self.solve_default(gs, fs)
        self.calc_strain(density)
        self.calc_stress(density)

    def calc_stress(self, x):
        # https://link.springer.com/article/10.1007/s11081-021-09675-3#Sec9
        q = 0.5  # 𝑞 is the stress relaxation parameter
        p = 8  # 𝑝 is the norm aggregation - higher values of p are more accurate but can cause oscillation (formula 8)
        D, B = self.material.elastic_matrix, self.material.strain_matrix

        flat_x = x.flatten('F')
        # calculate stress
        for i in range(0, x.size):  # todo: does stress and vm need to be flat?
            # relaxed stress vector - formulae 3-5
            # relaxed is meant to prevent singularity
            temp = flat_x[i] ** q * (D @ B @ self.displacement[self.element_dof_mat[i, :]]).T[0]
            self.stress[i, :] = temp
            # formula 7
            self.von_mises_stress[i] = np.sqrt(0.5 * ((temp[0] - temp[1]) ** 2 + (temp[0] - temp[2]) ** 2 + (temp[1] - temp[2]) ** 2 + 6 * sum(temp[3:6] ** 2)))

        self.total_stress = (self.von_mises_stress ** p).sum() ** (1/p)

    def calc_strain(self, density):
        self.strain = (
                    np.dot(self.displacement[self.element_dof_mat].reshape(self.total_nodes, 24), self.material.element_stiffness) *
                    self.displacement[self.element_dof_mat].reshape(self.total_nodes, 24)).sum(1).reshape(density.shape,
                                                                                                 order="F")
        self.compliance = (self.material.stiffness(density).flatten() * self.strain.flatten()).sum()

    @staticmethod
    def solve_default(stiffness, forces):
        return linalg.spsolve(stiffness, forces)

    @staticmethod
    def solve_iterative(stiffness, force):  # fixme
        # https://www.top3d.app/tutorials/iterative-solver-top3d
        d1 = stiffness.diagonal()
        precondition_matrix = diags(d1)
        linalg.cg(stiffness, force, tol=1e-8, maxiter=8000, M=precondition_matrix)


class Optimizer:
    def __init__(self, shape: typing.Tuple[int, int, int]):
        y, x, z = self.shape = shape
        self.total_nodes = x * y * z
        self.dc = np.ones(shape)
        self.dv = Filter.smoothen(np.ones(self.shape))  # in the paper this goes in sensistivity analysis, but it looks constant
        self.strain_energy = np.ones(shape)  # the amount of energy causes by compliance and stiffness
        self.change = 0

    def minimize_compliance(self, density: np.ndarray, model: FEA, volfrac: float) -> typing.Tuple[np.ndarray, float]:
        self.compliance_sensitivity(density, model)
        return self.optimality_criteria(density, self.dc, self.dv, volfrac)

    def compliance_sensitivity(self, density: np.ndarray, model: FEA):
        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS - figure 17-28 (minimum compliance)
        self.dc[:] = (-model.material.gradient(density).flatten() * model.strain.flatten()).reshape(density.shape)
        # Density Filtering (prevents irregular gaps)
        self.dc[:] = Filter.smoothen(self.dc)
        return self.dc, self.dv, model.compliance

    def stress_sensitivity(self, density: np.ndarray, model: FEA):
        p = 8
        q = 0.5
        # https://link.springer.com/article/10.1007/s11081-021-09675-3#ref-CR35
        strain_matrix, elastic_matrix = model.material.strain_matrix, model.material.elastic_matrix

        DvmDrs = np.zeros((model.total_nodes, 6))  # derivative of von mies stress with respect to relaxed stress
        DpnDvm = (model.von_mises_stress ** p).sum() ** (1 / p - 1)  # derivative of p normal stress (obj) with respect to von mises stress
        index_matrix = model.element_dof_mat.T
        for i in range(0, model.total_nodes):
            DvmDrs[i, 0] = 1/2 / model.von_mises_stress[i] * (2 * model.stress[i, 0] - model.stress[i, 1] - model.stress[i, 2])
            DvmDrs[i, 1] = 1/2 / model.von_mises_stress[i] * (2 * model.stress[i, 1] - model.stress[i, 0] - model.stress[i, 2])
            DvmDrs[i, 2] = 1/2 / model.von_mises_stress[i] * (2 * model.stress[i, 2] - model.stress[i, 0] - model.stress[i, 1])
            DvmDrs[i, 3] = 3 / model.von_mises_stress[i] * model.stress[i, 3]
            DvmDrs[i, 4] = 3 / model.von_mises_stress[i] * model.stress[i, 4]
            DvmDrs[i, 5] = 3 / model.von_mises_stress[i] * model.stress[i, 5]

        # calculation of T1
        # sum(dPNdVM[dVMds.T * dNdx * s]) - beta is between the square brackets
        beta = np.zeros((model.total_nodes, 1))
        for i in range(0, model.total_nodes):
            element_displacement = model.displacement[model.element_dof_mat[i, :], :].T.reshape((24, 1), order="F")
            DnDx = q * (density.flatten('F')[i]) ** (q - 1)  # Eq. 19 - https://link.springer.com/article/10.1007/s11081-021-09675-3#Equ19
            DsDx = elastic_matrix @ strain_matrix @ element_displacement  # Eq. 21  - https://link.springer.com/article/10.1007/s11081-021-09675-3#Equ21
            # todo: where does the p-1 come from
            beta[i] = DnDx * model.von_mises_stress[i] ** (p - 1) * DvmDrs[i, :] @ DsDx  # fixme is it (@ or *)?
        T1 = DpnDvm * beta

        # calculation of T2
        # 𝛾 [gama] = sum(n(x) * dPNdVM * dSdX.T * dVMdRS)
        gama = np.zeros(model.displacement.shape)
        for i in range(0, model.total_nodes):
            index = index_matrix[:, i]
            DsDx = elastic_matrix @ strain_matrix
            n = density.flatten('F')[i] ** q
            update = n * DpnDvm * DsDx.T @ DvmDrs[i, :].T * model.von_mises_stress[i] ** (p - 1)
            gama[index] = gama[index] + update.reshape((24, 1))
        # K𝜆 [lambda] = 𝛾 * K.inv()
        lamda = np.zeros(model.displacement.shape)
        lamda[model.free_dofs, 0] = model.solve_default(model.global_stiffness[model.free_dofs, :][:, model.free_dofs], gama[model.free_dofs, :])

        T2 = np.zeros((model.total_nodes, 1))
        for i in range(0, model.total_nodes):
            index = index_matrix[:, i]
            # T2 = -𝜆.T * dKdx * U [Eq. 29]
            dKdX = model.material.penal * density.flatten('F')[i] ** (model.material.penal - 1) * model.material.element_stiffness
            T2[i] = -lamda[index].T @ dKdX @ model.displacement[index]

        DpnDx = T1 + T2
        return DpnDx.reshape(density.shape, order="F")

    def optimality_criteria(self, x: np.ndarray, dc: np.ndarray, dv: np.ndarray, volfrac: float) -> typing.Tuple[np.ndarray, float]:
        """
        Applies the gradients from the sensitivity analysis to update the material distribution
        :param x: The density values of the array
        :param dc: The derivative of compliance with respect to density
        :param dv: Some sorta volume derivative
        :param volfrac: the target percentage of the volume you want to be filled
        :return: The distribution of material of the structure, the max change
        """
        # OPTIMALITY CRITERIA UPDATE
        l1 = 0
        l2 = 1e9
        move = 0.2
        xnew = np.zeros(x.shape)
        density = np.zeros(x.shape)
        # looks like this imposes the volume constraint
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # max and min constrict between 0 and 1, move also limits the change
            xnew[:] = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
            xnew[:] = coerce_elements(xnew)
            #density[:] = Filter.smoothen(xnew)
            f = Filter.instance
            density[:] = np.array(f.weights * xnew.flatten("F") / f.net_weight.flatten("F")).reshape(density.shape, order="F")
            if density.mean() > volfrac:  #
                l1 = lmid
            else:
                l2 = lmid
        change = abs(xnew - x).max()
        return xnew, change


class LoadCase:
    net_importance = 0

    def __init__(self, shape, importance):
        self.importance = importance
        self.net_importance += importance
        y_nodes, x_nodes, z_nodes = self.shape = shape
        self.num_dofs = (x_nodes + 1) * (y_nodes + 1) * (z_nodes + 1)  # number of degrees of freedom
        self._forces = np.zeros((self.num_dofs, 3))
        self._dof_freedom = np.ones((self.num_dofs, 3), dtype=bool)

    @property
    def force(self): return self._forces.reshape((self.num_dofs*3, 1))
    @property
    def free_dof(self): return np.where(self._dof_freedom.flatten() > 0)[0]
    @property
    def relative_importance(self): return self.importance / self.net_importance

    def add_force(self, force, mask):
        self._forces[self._get_dof(mask)] = force
        return self

    def affix(self, mask):
        self._dof_freedom[self._get_dof(mask)] = False
        return self

    def _get_dof(self, mask):
        y, x, z = mask.shape
        i2, i1, i3 = np.where(mask > 0)
        load = i3 * x * y + i1 * y + (y - 1 - i2)
        return load


def coerce_elements(x: np.ndarray) -> np.ndarray:
    '''
    This is where you can implement active and passive elements (ie requirements for each nodes are solid and empty)
    Just rewrite the array at the specific indices to the values they need to be
    :param x: The unmodified structure
    :return: The modified structure
    '''
    # https://www.top3d.app/tutorials/active-and-passive-elements-top3d
    return x  # current implementation has no active or passive elements


def display_3d(structure: np.ndarray, strain: np.ndarray):
    structure = np.swapaxes(np.flip(np.swapaxes(structure, 0, 2), 2), 0, 1)  # reorientate the structure
    strain = np.swapaxes(np.flip(np.swapaxes(strain, 0, 2), 2), 0, 1)
    shape = structure.shape
    y_nodes, x_nodes, z_nodes = shape
    strain = np.minimum(1.0, strain / strain[structure > 0.5].max())
    total_nodes = x_nodes * y_nodes * z_nodes
    hue = 2 / 3 - strain * 2 / 3  # get red to blue hue depending on displacement
    saturation = np.ones(shape)  # always high saturation
    value = np.ones(shape)  # always bright
    hsv = np.stack((hue, saturation, value), axis=3)  # build color
    rgb = hsv_to_rgb(hsv.reshape((total_nodes, 3))).reshape((*shape, 3))  # convert to accepted format
    alpha = structure.reshape((*shape, 1))
    rgba = np.concatenate((rgb, alpha), axis=3)  # same thing with tranparency equal to density
    ax = plt.figure().add_subplot(projection='3d')  # todo: look into interactivity so you can look around
    ax.set_box_aspect((y_nodes, x_nodes, z_nodes))
    blocks = np.zeros(structure.shape, dtype=bool)
    blocks[structure > 0.5] = True
    # blocks[0, 0, 4] = True
    ax.voxels(blocks, facecolors=rgba)
    plt.ion()
    plt.show()


def save(structure: np.ndarray, filename: str) -> None: np.save(filename+".npy", structure)
def load(filename: str) -> np.ndarray: return np.load(filename+".npy")


def main(x_nodes: int, y_nodes: int, z_nodes: int, volfrac: float, penal: float, rmin: float):
    shape = (y_nodes, x_nodes, z_nodes)  # the shape of our grid - used to create arrays
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(shape, dtype=float)  # start as a homogenous solid
    density = x.copy()

    # prepare helper classes
    material = Material(0.3, 1.0, 1e-19, penal)  # define the material properties of you structure
    modeler = FEA(shape, material)  # setup FEA analysis
    f = Filter(rmin, shape)  # filter to prevent gaps

    # Set loop counter and gradient vectors
    change = 1
    optimizer = Optimizer(shape)  # the class that contains the optimization algorithm
    start_time = time.time()
    for loop in range(2000):
        if change < 0.01: break  # if you have reached the minimum its not worth continuing
        # Setup and solve FE problem
        modeler.displace(density)
        # Objective and sensitivity
        compliance_gradient, volume_gradient, obj = optimizer.compliance_sensitivity(density, modeler)
        ds = optimizer.stress_sensitivity(density, modeler)
        # Optimality criteria
        x[:], change = optimizer.optimality_criteria(x, compliance_gradient, volume_gradient, volfrac)
        # Filter design variables
        density[:] = np.array(f.weights * x.flatten("F") / f.net_weight.flatten("F")).reshape(density.shape, order="F") # smooth_filter(x)

        # Write iteration history to screen (req. Python 2.6 or newer)
        print(f"i: {loop} (avg {round((time.time()-start_time)/(loop+1), 2)} sec),\t"
              f"comp.: {round(modeler.compliance, 3)}\t"
              f"Vol.: {round(density.mean(), 3)*100}%,\t"
              f"ch.: {round(change, 2)}")
    display_3d(x, modeler.strain)
    save(x, "test")


def run_load():
    structure = load("test")
    shape = (y_nodes, x_nodes, z_nodes) = structure.shape
    num_dofs = 3 * (x_nodes + 1) * (y_nodes + 1) * (z_nodes + 1)  # number of degrees of freedom - format is [Fx1, Fy1, Fz1, Fx2 ... Fz#]
    displacements = np.zeros((num_dofs, 1))
    material = Material(0.3, 1.0, 1e-19, 4)  # define the material properties of you structure
    modeler = FEA(shape, material)  # setup FEA analysis
    smooth_filter = Filter(1.5, shape)  # filter to prevent gaps
    optimizer = Optimizer(shape)  # the class that contains the optimization algorithm
    start_time = time.time()
    # Setup and solve FE problem
    modeler.displace(structure)
    # Objective and sensitivity
    optimizer.compliance_sensitivity(structure, modeler)
    display_3d(structure, modeler.von_mises_stress.reshape(structure.shape, order="F"))


if __name__ == '__main__':
    # run_load()
    main(10, 10, 10, 0.3, 4, 2.5)
