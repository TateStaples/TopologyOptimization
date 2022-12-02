__author__ = "Tate Staples"

import typing
from scipy import sparse
from materials import Material
from load import LoadCase
import numpy as np

# Hyperparameters
stress_relaxation = 0.1  # 𝑞 is the stress relaxation parameter - prevent singularity of stress at low density
norm_aggregation = 10  # 𝑝 is the norm aggregation - higher p estimates max stress better but too high can be unstable


class FEA:
    """
    Class the performs the physical analysis of how the structure deforms
    Equations:
    1. force [F] = global stiffness [K] * displacement[U]  -> FEA approximates structure as points connected by springs
    2. stress [s] @ elem i = solid stiffness [D] * strain matrix [B] @ elem i * U @ elem i = (σ𝑖𝑥,σ𝑖𝑦,σ𝑖𝑧,σ𝑖𝑥𝑦,σ𝑖𝑦𝑧,σ𝑖𝑧𝑥).𝑇
    3. relaxed stress(x) [rs/σ] = 𝜂(𝑥) * s, where 𝜂(𝑥) = x^q for some q that prevents singularity from low densities
    4. von mises stress[vm] = sqrt(σ𝑥^2+σ𝑦^2+σ𝑧^2−σ𝑥σ𝑦−σ𝑦σ𝑧−σ𝑧σ𝑥+3t𝑥𝑦^2+3t𝑦𝑧^2+3t𝑧𝑥^2) -> *magnitude of stress
    5. global p norm stress [pn] = sum(vm^p)^(1-p) -> estimated max stress - large p better but can oscillate (p>30)
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
        self.forces = load_case.force
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
        self.global_stiffness = sparse.csc_matrix((self.num_dofs, self.num_dofs))
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
        self.global_stiffness = sparse.coo_matrix((elementwise_stiffness, (self.iK, self.jK)), shape=(self.num_dofs, self.num_dofs)).tocsc()
        self.global_stiffness = (self.global_stiffness + self.global_stiffness.T) / 2.0  # average with previous?
        # Remove constrained dofs from matrix
        gs = self.global_stiffness[self.free_dofs, :][:, self.free_dofs]
        fs = self.forces[self.free_dofs, :]
        # solves F = KU (eq. 1) for U, either by doing K.inv() * F or some iterative solver
        # self.displacement[self.free_dofs, 0], _ = self.solve_iterative(gs, fs)
        self.displacement[self.free_dofs, 0] = self.solve(gs, fs)

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
            n = flat_x[i] ** stress_relaxation  # 𝜂(𝑥) from eq. 3
            s = (self.material.elastic_matrix @ self.material.strain_matrix @ self.displacement[self.element_dof_mat[i, :]]).T[0]  # eq. 2
            relaxed_stress = n * s  # eq. 3
            self.stress[i, :] = relaxed_stress
            self.von_mises_stress[i] = np.sqrt(0.5 * ((relaxed_stress[0] - relaxed_stress[1]) ** 2 + (relaxed_stress[0] - relaxed_stress[2]) ** 2 + (relaxed_stress[1] - relaxed_stress[2]) ** 2 + 6 * sum(relaxed_stress[3:6] ** 2)))  # eq. 4
        # print(self.von_mises_stress.max())
        self.max_stress = (self.von_mises_stress ** norm_aggregation).sum() ** (1 / norm_aggregation)  # eq. 5

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

    def solve(self, stiff, force):
        """
        Solve the equation with a method depending on complexity of the equation
        :return: solved displacements
        """
        if self.total_nodes < 1000:
            return self.solve_default(stiff, force)
        else:
            return self.solve_iterative(stiff, force)[0]

    @staticmethod
    def solve_default(stiffness, forces):
        """
        Solve the Equation F = KU for U [displacement] This is an expanded version of Hooke's law F = k * ∆x
        This is a standard solver that computes U by solving for the K inverse which can slow at larger problems
        :param stiffness: the global stiffness matrix (at free dofs)
        :param forces: the forces being applied to the structure (at free dofs)
        :return: the nodel displacemetns (at freedofs)
        """
        return sparse.linalg.spsolve(stiffness, forces)

    def solve_iterative(self, stiffness, force):
        """
        Solve the Equation F = KU for U [displacement] This is an expanded version of Hooke's law F = k * ∆x
        This is a iterative solver that approximates U using the conjugate gradient method (ie weird math)
        source: https://www.top3d.app/tutorials/iterative-solver-top3d
        :param stiffness: the global stiffness matrix (at free dofs)
        :param force: the forces being applied to the structure (at free dofs)
        :return: the nodel displacements (at freedofs)
        """
        precondition_matrix = sparse.diags(1/stiffness.diagonal())
        # return linalg.cg(stiffness, force, x0=self.displacement[self.free_dofs, :], tol=1e-8, maxiter=8000, M=precondition_matrix)
        return sparse.linalg.cg(stiffness, force, x0='Mb', tol=1e-8, maxiter=8000, M=precondition_matrix)