import matplotlib
import numpy as np
import math
from scipy.sparse import *
import typing
import matplotlib.pyplot as plt
from scipy.sparse import linalg
from matplotlib.colors import hsv_to_rgb
import time
import gyroidizer
import nlopt


class Material:
    """
    Describes the material properties you are using
    1. stiffness[E] (density[x]) at elem i = min stiffness + x^p * (max stiffness[E0] − min stiffness[Emin]) -> SIMP method
    2. ∫∫∫ strain_matrix.T [B] * unit elastic matrix [D] * B 𝑑x𝑑y𝑑z
    3. global stiffness = sum(E(x) * element stiffness)
    """
    def __init__(self, poisson: float = 0.3, max_stiff: float = 1.0, min_stiff: float = 1e-19, penal: float = 3):
        """
        Initialize the properties of for element
        :param poisson: the deformation shrinkage (for x unit strech, how much y pinch)
        :param max_stiff: Young's modulus of solid material (stiffness)
        :param min_stiff: Young's modulus of void-like material
        """
        assert poisson > 0 and min_stiff > 0
        self.poisson = poisson
        self.max_stiff = max_stiff
        self.min_stiff = min_stiff
        self.element_stiffness = self._element_stiffness()
        self.strain_matrix = self._strain_matrix()
        self.elastic_matrix = self._elastic_matrix()
        self.penal = penal

    def _element_stiffness(self) -> np.ndarray:
        """
        Converts the material properties into a matrix representing the stiffness of a hexahedral (cube) element
        :return: 24x24 array representing the x,y,z stiffness of each corner of the element
        """
        # source: https://link.springer.com/article/10.1007/s11081-021-09675-3#ref-CR23
        # solved formula for eq 2
        A = np.array([
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
            [K1,    K2,     K3,     K4],
            [K2.T,  K5,     K6,     K3.T],
            [K3.T,  K6,     K5.T,   K2.T],
            [K4,    K3,     K2,     K1.T]
        ])
        return KE.swapaxes(1, 2).reshape(24, 24)

    def _strain_matrix(self) -> np.ndarray:
        """
        Gets the strain matrix for a hexahedral element
        :return: 6x24 matrix
        """
        # https://link.springer.com/article/10.1007/s00158-019-02323-6
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

    def _elastic_matrix(self) -> np.ndarray:
        """
        Get the elasticity of a hexahedral element based on poisson ratio
        https://link.springer.com/article/10.1007/s11081-021-09675-3#Sec5 - Eq. 10
        :return: 6x6 matrix
        """
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

    def stiffness(self, density: np.ndarray) -> np.ndarray:
        """
        Get youngs modulus based on the Solid Isotropic Material with Penalization method
        :param density: density distribution in structure
        :return: the youngs modulus at each node
        """
        # https://link.springer.com/article/10.1007/s11081-021-09675-3#Sec5 - Eq. 13
        return self.min_stiff + density ** self.penal * (self.max_stiff - self.min_stiff)

    def gradient(self, density: np.ndarray) -> np.ndarray:
        """
        Gradient of the youngs modulus with respect to density
        :param density: density distribution in nodes
        :return: the gradient of the stiffness compared to density (same shape as density)
        """
        return self.penal * density ** (self.penal - 1) * (self.max_stiff - self.min_stiff)  # see equation 26 of matlab paper


class Gyroid(Material):
    """
    Modified version of Material that reflects the scaling law of gyroids
    """
    def stiffness(self, density: np.ndarray) -> np.ndarray:
        # https://link.springer.com/article/10.1007/s00170-020-06542-w
        return (-482.65 * np.power(density, 3) + np.power(density, 2) + 27.693 * density + self.min_stiff) * self.max_stiff / 3145e6

    def gradient(self, density: np.ndarray) -> np.ndarray:
        return (3 * -482.65 * np.power(density, 2) + np.power(density, 2) + 27.693 * density + self.min_stiff) * self.max_stiff / 3145e6  # ratio of titanium to PLA


class Filter:  # fixme: move this out of a class and into a new file
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


class LoadCase:
    """
    Simplified interface for applying load and affixing portions of the structure
    """
    net_importance = 0

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

    @staticmethod
    def bridge(shape: typing.Tuple[int, int, int]):
        y_nodes, x_nodes, z_nodes = shape
        force_location = np.zeros((y_nodes + 1, (x_nodes + 1), (z_nodes + 1)), dtype=bool)
        fix_location = force_location.copy()
        force_location[0, x_nodes, :] = True
        fix_location[:, 0, :] = True
        load_case = LoadCase(shape, 1).add_force((0, -1, 0), force_location).affix(fix_location)
        return load_case

    @staticmethod
    def table(shape: typing.Tuple[int, int, int]):
        y_nodes, x_nodes, z_nodes = shape
        force_location = np.zeros((y_nodes + 1, (x_nodes + 1), (z_nodes + 1)), dtype=bool)
        fix_location = force_location.copy()
        force_location[y_nodes, :, :] = True
        fix_location[0, :, :] = True
        load_case = LoadCase(shape, 1).add_force((0, 1, 0), force_location).affix(fix_location, lock_x=False, lock_z=False)
        return load_case

    @staticmethod
    def bone(shape: typing.Tuple[int, int, int]):
        y_nodes, x_nodes, z_nodes = shape
        force_location = np.zeros((y_nodes + 1, (x_nodes + 1), (z_nodes + 1)), dtype=bool)
        fix_location = force_location.copy()
        f2 = force_location.copy()
        force_location[y_nodes, :, :] = True
        fix_location[y_nodes//2, :, :] = True
        f2[0, :, :] = True
        load_case = LoadCase(shape, 1).add_force((0, -8000, 0), force_location).add_force((0, 8000, 0), f2)\
            .affix(fix_location, lock_x=False, lock_z=False)
        return load_case


class FEA:
    """
    Class the performs the physical analysis of how the structure deforms
    Equations:
    1. force [F] = global stiffness [K] * displacement[U]
    2. stress [s] at elem i = solid stiffness [D] * strain matrix [B] at elem i * U at elem i = (σ𝑖𝑥,σ𝑖𝑦,σ𝑖𝑧,σ𝑖𝑥𝑦,σ𝑖𝑦𝑧,σ𝑖𝑧𝑥).𝑇
    3. relaxed stress(x) [rs/σ] = 𝜂(𝑥) * s, where 𝜂(𝑥) is some penalization scheme -> x^q for some q that prevents singularity from low densities
    4. von mises stress[vm] = sqrt(σ𝑥^2+σ𝑦^2+σ𝑧^2−σ𝑥σ𝑦−σ𝑦σ𝑧−σ𝑧σ𝑥+3t𝑥𝑦^2+3t𝑦𝑧^2+3t𝑧𝑥^2) -> magnitude of stress for comparing to yield
    5. global p norm stress [pn] = sum(vm^p)^(1-p) -> greater p is a more accurate but can cause oscillation (p>30)
    """
    def __init__(self, shape: typing.Tuple[int, int, int], material: Material, load_case: LoadCase):
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

        self.max_stress = 0
        self.von_mises_stress = np.zeros((self.total_nodes, 1))  # von Mises stress vector
        self.stress = np.zeros((self.total_nodes, 6))  # stress has 6 components for each

    def displace(self, density: np.ndarray):
        """
        Applies the load condition to the supplied structure
        :param density: the denisty field of the structure
        :return: None. Displacements and other phyiscal properties are internally updated
        """
        # https://link.springer.com/article/10.1007/s11081-021-09675-3#Sec5 - Eq. 12
        # eq. material.3
        elementwise_stiffness = ((self.material.element_stiffness.flatten()[np.newaxis]).T*self.material.stiffness(density).flatten('F')).flatten(order='F')
        self.global_stiffness = coo_matrix((elementwise_stiffness, (self.iK, self.jK)), shape=(self.num_dofs, self.num_dofs)).tocsc()
        self.global_stiffness = (self.global_stiffness + self.global_stiffness.T) / 2.0  # average with previous?
        # Remove constrained dofs from matrix
        gs = self.global_stiffness[self.free_dofs, :][:, self.free_dofs]
        fs = self.forces[self.free_dofs, :]
        # solves F = KU (eq. 1) for U, either by doing K.inv() * F or some iterative solver
        # self.displacement[self.free_dofs, 0], _ = self.solve_iterative(gs, fs)
        self.displacement[self.free_dofs, 0] = self.solve_default(gs, fs)

    def calc_stress(self, x):
        """
        Calculate the stresses caused by the deformations of the laod
        :param x: the structure densities
        :return: None. Internally updates stress, VM stress, and max_stress
        """
        # https://link.springer.com/article/10.1007/s11081-021-09675-3#Sec9
        flat_x = x.flatten('F')
        # calculate stress
        for i in range(0, x.size):  # fixme: does stress and vm need to be flat?
            n = flat_x[i] ** q  # 𝜂(𝑥) from eq. 3
            s = (self.material.elastic_matrix @ self.material.strain_matrix @ self.displacement[self.element_dof_mat[i, :]]).T[0]  # eq. 2
            relaxed_stress = n * s  # eq. 3
            self.stress[i, :] = relaxed_stress
            self.von_mises_stress[i] = np.sqrt(0.5 * ((relaxed_stress[0] - relaxed_stress[1]) ** 2 + (relaxed_stress[0] - relaxed_stress[2]) ** 2 + (relaxed_stress[1] - relaxed_stress[2]) ** 2 + 6 * sum(relaxed_stress[3:6] ** 2)))  # eq. 4

        self.max_stress = (self.von_mises_stress ** p).sum() ** (1 / p)  # eq. 5

    def calc_strain(self, density):
        """
        Runs the calculation of the strain caused by the load deformation
        :param density: the densities of the structure
        :return: None. Strain and compliance are internally updated
        """
        self.strain = (
                    np.dot(self.displacement[self.element_dof_mat].reshape(self.total_nodes, 24), self.material.element_stiffness) *
                    self.displacement[self.element_dof_mat].reshape(self.total_nodes, 24)).sum(1).reshape(density.shape,order="F")
        self.compliance = (self.material.stiffness(density).flatten() * self.strain.flatten()).sum()

    @staticmethod
    def solve_default(stiffness, forces):
        """
        Solve the Equation F = KU for U [displacement] This is an expanded version of Hooke's law F = k * ∆x
        This is a standard solver that computes U by solving for the K inverse which can slow at larger problems
        :param stiffness: the global stiffness matrix (at free dofs)
        :param forces: the forces being applied to the structure (at free dofs)
        :return: the nodel displacemetns (at freedofs)
        """
        return linalg.spsolve(stiffness, forces)

    def solve_iterative(self, stiffness, force):
        """
        Solve the Equation F = KU for U [displacement] This is an expanded version of Hooke's law F = k * ∆x
        This is a iterative solver that approximates U using the conjugate gradient method (ie weird math)
        source: https://www.top3d.app/tutorials/iterative-solver-top3d
        :param stiffness: the global stiffness matrix (at free dofs)
        :param force: the forces being applied to the structure (at free dofs)
        :return: the nodel displacemetns (at freedofs)
        """
        precondition_matrix = diags(stiffness.diagonal())
        #return linalg.cg(stiffness.todense(), force.todense(), x0=self.displacement[self.free_dofs, :], tol=1e-3, maxiter=1000, M=precondition_matrix)
        return linalg.cg(stiffness.todense(), force.todense(), x0=self.displacement[self.free_dofs, :], tol=1e-8, maxiter=8000, M=precondition_matrix)


class SensitivityAnalysis:
    """ Calculator for finding the important gradients to help in optimization """
    def __init__(self, shape: typing.Tuple[int, int, int]):
        y, x, z = self.shape = shape
        self.total_nodes = x * y * z
        self.dc = np.ones(shape)
        self.dv = Filter.smoothen(np.ones(self.shape))  # in the paper this goes in sensistivity analysis, but it looks constant

    def compliance_sensitivity(self, density: np.ndarray, model: FEA):
        """
        Calculate the gradient of compliance with respect to density (dCdX)
        :param density: densities of the structure
        :param model: the physical model of the structure. Contains important properties
        :return: Compliance gradient, Volume gradient, current compliance
        """
        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS - figure 17-28 (minimum compliance)
        self.dc[:] = -model.material.gradient(density) * model.strain  # i dont actually know where this is derived in the paper
        # Density Filtering (prevents irregular gaps)
        self.dc[:] = Filter.smoothen(self.dc)
        return self.dc, self.dv, model.compliance

    def stress_sensitivity(self, density: np.ndarray, model: FEA):
        """
        Calculate the gradient of the max stress with respect to density. The derivation of this is really weird so beware
        source: https://link.springer.com/article/10.1007/s11081-021-09675-3#ref-CR35

        Equations:
        dPNdVM = sum(stress^p)^(1/p-1) * vm^(p-1) -> deriv of FEA.5
        dNdX = q*x^(q-1) -> n(x) = x^q
        dVMdRS = linear: 0.5 * 1/vm (sx - sy - sz), shear: 3 / vm * τ -> deriv of FEA. 4
        𝛽 [beta]  = vm^(p-1) * dVMdRS.T * dNdX * rs
        T1 = sum(dPNdVM * 𝛽)

        𝛾 [gama] = sum(n(x) * dPNdVM * dSdX.T * dVMdRS)
        K𝜆 [lambda] = 𝛾 * K.inv()
        dKdX = L.T * material.gradient(x) * element_stiffness * L
        U = F/K
        T2 = -𝜆.T * dKdX * U

        dPNdX = T1 + T2

        :param density: the densities of the structure
        :param model: physical model
        :return: gradient of the max stress with respect to density
        """
        strain_matrix, elastic_matrix = model.material.strain_matrix, model.material.elastic_matrix

        DvmDrs = np.zeros((self.total_nodes, 6))  # derivative of von mies stress with respect to relaxed stress
        DpnDvm = (model.von_mises_stress ** p).sum() ** (1 / p - 1)  # derivative of p normal stress (obj) with respect to von mises stress
        index_matrix = model.element_dof_mat.T
        for i in range(0, self.total_nodes):
            DvmDrs[i, 0] = 1/2 / model.von_mises_stress[i] * (2 * model.stress[i, 0] - model.stress[i, 1] - model.stress[i, 2])
            DvmDrs[i, 1] = 1/2 / model.von_mises_stress[i] * (2 * model.stress[i, 1] - model.stress[i, 0] - model.stress[i, 2])
            DvmDrs[i, 2] = 1/2 / model.von_mises_stress[i] * (2 * model.stress[i, 2] - model.stress[i, 0] - model.stress[i, 1])
            DvmDrs[i, 3] = 3 / model.von_mises_stress[i] * model.stress[i, 3]
            DvmDrs[i, 4] = 3 / model.von_mises_stress[i] * model.stress[i, 4]
            DvmDrs[i, 5] = 3 / model.von_mises_stress[i] * model.stress[i, 5]

        # calculation of T1
        # sum(vm^(p-1) * dPNdVM[dVMds.T * dNdx * s]) - beta is between the square brackets - Eq. 20
        beta = np.zeros((self.total_nodes, 1))
        for i in range(0, self.total_nodes):
            element_displacement = model.displacement[model.element_dof_mat[i, :], :].T.reshape((24, 1), order="F")
            DnDx = q * (density.flatten('F')[i]) ** (q - 1)  # Eq. 19 - https://link.springer.com/article/10.1007/s11081-021-09675-3#Equ19
            DsDx = elastic_matrix @ strain_matrix @ element_displacement  # Eq. 21  - https://link.springer.com/article/10.1007/s11081-021-09675-3#Equ21
            beta[i] = DnDx * model.von_mises_stress[i] ** (p - 1) * DvmDrs[i, :] @ DsDx
        T1 = DpnDvm * beta

        # calculation of T2
        # 𝛾 [gama] = sum(n(x) * dPNdVM * dSdX.T * dVMdRS)=
        gama = np.zeros(model.displacement.shape)
        for i in range(0, self.total_nodes):
            index = index_matrix[:, i]
            DsDx = elastic_matrix @ strain_matrix
            n = density.flatten('F')[i] ** q
            update = n * DpnDvm * DsDx.T @ DvmDrs[i, :].T * model.von_mises_stress[i] ** (p - 1)
            gama[index] = gama[index] + update.reshape((24, 1))
        # K𝜆 [lambda] = 𝛾 * K.inv()
        lamda = np.zeros(model.displacement.shape)
        lamda[model.free_dofs, 0] = model.solve_default(model.global_stiffness[model.free_dofs, :][:, model.free_dofs], gama[model.free_dofs, :])

        T2 = np.zeros((self.total_nodes, 1))
        for i in range(0, self.total_nodes):
            index = index_matrix[:, i]
            # T2 = -𝜆.T * dKdx * U [Eq. 29]
            dKdX = model.material.gradient(density).flatten('F')[i] * model.material.element_stiffness
            T2[i] = -lamda[index].T @ dKdX @ model.displacement[index]

        DpnDx = T1 + T2
        return DpnDx.reshape(density.shape, order="F")


class Parameter:
    """Class to wrap value and gradient for objective and constraint values"""
    def __init__(self, value, gradient, maximum=None): self.value = value; self.gradient = gradient; self.max = maximum


class Optimizer:
    def __init__(self, shape, update):
        x, y, z = self.shape = shape
        self.total_nodes = n = x * y * z  # number of design variables
        self.min_densities = 1e-3 * np.ones(n)  # minimum values for design
        self.max_densities = np.ones(n)  # max values for design
        self.opt = opt = nlopt.opt(nlopt.LD_MMA, n)
        self.updater = update
        self.results = None
        self.change = 1
        self.prev = np.ones(n)
        self.iteration = 1
        self.start_time = time.time()
        opt.set_lower_bounds(self.min_densities)
        opt.set_upper_bounds(self.max_densities)
        opt.add_inequality_constraint(self.get_volume)
        opt.add_inequality_constraint(self.get_stress)
        opt.set_min_objective(self.get_compliance)
        opt.set_ftol_rel(1e-4)

    def optimize(self, x):
        return self.opt.optimize(x)

    def done(self):
        return self.iteration > 2000 or self.change < 0.01

    def get_compliance(self, x, grad):
        self.change = abs(x-self.prev).max()
        self.results = next(self.updater(x.reshape(self.shape, order='F')))
        comp, vol, stress = self.results
        if grad.size > 0: grad[:] = comp.gradient.flatten('F')
        print(f"i: {self.iteration} ({round((time.time() - self.start_time), 2)}s),\t"
              f"comp.: {round(comp.value)}\t"
              f"stress: {round(stress.value)}\t"
              f"vol.: {round(vol.value * 100, 1)}%,\t"
              f"ch.: {round(self.change, 2)}")
        self.prev[:] = x.copy(); self.iteration += 1; self.start_time = time.time()
        return comp.value

    def get_volume(self, x, grad):
        _, vol, _ = self.results
        if grad.size > 0: grad[:] = vol.gradient.flatten('F')
        return vol.value - vol.max

    def get_stress(self, x, grad):
        _, _, stress = self.results
        if grad.size > 0: grad[:] = stress.gradient.flatten('F')
        return stress.value - stress.max


class Display:
    def __init__(self, shape):
        # matplotlib.use("TkAgg")
        (y_nodes, x_nodes, z_nodes) = shape
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_box_aspect((x_nodes, z_nodes, y_nodes))
        plt.ion()
        self.shapes = list()

    def display_3d(self, structure: np.ndarray, strain: np.ndarray):
        self.shapes.append((structure, strain))
        structure = np.swapaxes(np.flip(np.swapaxes(structure, 0, 2), 2), 0, 1)  # reorientate the structure
        strain = np.swapaxes(np.flip(np.swapaxes(strain, 0, 2), 2), 0, 1)
        shape = structure.shape
        y_nodes, x_nodes, z_nodes = shape
        strain = np.minimum(1.0, strain / strain[structure > 0.1].max())
        total_nodes = x_nodes * y_nodes * z_nodes
        hue = 2 / 3 - strain * 2 / 3  # get red to blue hue depending on displacement
        saturation = np.ones(shape)  # always high saturation
        value = np.ones(shape)  # always bright
        hsv = np.stack((hue, saturation, value), axis=3)  # build color
        rgb = hsv_to_rgb(hsv.reshape((total_nodes, 3))).reshape((*shape, 3))  # convert to accepted format
        alpha = structure.reshape((*shape, 1))
        rgba = np.concatenate((rgb, alpha), axis=3)  # same thing with tranparency equal to density
        # https://www.tutorialspoint.com/how-to-get-an-interactive-plot-of-a-pyplot-when-using-pycharm
        blocks = np.zeros(structure.shape, dtype=bool)
        blocks[structure > 0.1] = True
        # blocks[0, 0, 4] = True
        self.ax.clear()
        self.ax.voxels(blocks, facecolors=rgba)
        plt.draw()

    def _animate(self, frame):  # fixme: make colors consistent
        struct, strain = self.shapes[frame]
        self.display_3d(struct, strain)
        return self.ax

    def make_animation(self):
        from matplotlib.animation import FuncAnimation
        ani = FuncAnimation(self.fig, self._animate, frames=len(self.shapes), interval=100)
        # Save as gif
        ani.save('animation.gif', fps=5)
        plt.show()


def save(structure: np.ndarray, filename: str) -> None: np.save(filename+".npy", structure)
def load(filename: str) -> np.ndarray: return np.load(filename+".npy")


def main(x_nodes: int, y_nodes: int, z_nodes: int, volfrac: float, penal: float, rmin: float):
    shape = (y_nodes, x_nodes, z_nodes)  # the shape of our grid - used to create arrays
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(shape, dtype=float)  # start as a homogenous solid
    density = x.copy()

    # prepare helper classes
    """
    Defining units:
    - Force: Newtons (N)
    - Displacement: meters (m)
    - Stiffness: N * m
    - Stress: N / m^2 (ie Pascals (pa))
    """
    material = Material(0.35, 119e9, 1e-19, penal)  # define the material properties of you structure
    modeler = FEA(shape, material, LoadCase.bone(shape))  # physics simulations
    f = Filter(rmin, shape)  # filter to prevent gaps
    sens = SensitivityAnalysis(shape)  # find the gradients
    yield_stress = 730e6  # base titanium yield in 380 MPa
    d = Display(shape)
    # Set loop counter and gradient vectors
    change = 1
    start_time = time.time()
    def update(density):
        modeler.displace(density)
        modeler.calc_strain(density)
        modeler.calc_stress(density)
        # Objective and sensitivity
        compliance_gradient, volume_gradient, obj = sens.compliance_sensitivity(density, modeler)
        ds = sens.stress_sensitivity(density, modeler)
        compliance = Parameter(obj, compliance_gradient)
        volume = Parameter(x.mean(), volume_gradient, volfrac)
        stress = Parameter(modeler.max_stress, ds, yield_stress)
        yield compliance, volume, stress

    opt = Optimizer(shape, update)  # updates the structure to new distribution
    while not opt.done():
        x = opt.opt.optimize(x.flatten('F'))
    x = x.reshape(shape, order='F')
    modeler.calc_stress(x)
    # d.make_animation()
    d.display_3d(x, modeler.von_mises_stress.reshape(shape, order="F"))
    plt.show()
    gyroidizer.gyroidize(x)
    save(x, "test2")


def run_load():
    """
    Quickly load and display the last generated structure
    """
    structure = load("test2")
    gyroidizer.gyroidize(structure, scale=2, resolution=20j)
    quit()
    shape = (y_nodes, x_nodes, z_nodes) = structure.shape
    material = Gyroid(0.3, 1.0, 1e-19, 4)  # define the material properties of you structure
    modeler = FEA(shape, material, LoadCase.bridge(shape))  # setup FEA analysis
    smooth_filter = Filter(1.5, shape)  # filter to prevent gaps
    # Setup and solve FE problem
    modeler.displace(structure)
    # Objective and sensitivity
    modeler.calc_stress(structure)
    Display(shape).display_3d(structure, modeler.von_mises_stress.reshape(structure.shape, order="F"))
    plt.show()

if __name__ == '__main__':
    q = 0.5  # 𝑞 is the stress relaxation parameter - prevent singularity
    p = 15  # 𝑝 is the norm aggregation - higher values of p is closer to max stress but too high can cause oscillation and instability
    run_load()
    # main(8, 20, 8, 0.1, 3, 1.5)
