import numpy as np
import math
from scipy.sparse import *
import typing

from scipy.sparse.linalg import spsolve

display = False
# todo
# iterative solver
# active/passive elements
# gyroid material
# different optimization algorithms


class Material:
    """
    Class the wraps the material properties you are using.
    Used to calculate the stiffness given the density
    """
    def __init__(self, poisson: float=0.3, max_stiff: float = 1.0, min_stiff: float = 1e-19, penal: int = 3):
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
        self.penal = penal


    def _element_stiffness(self):
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

    def stiffness(self, density):
        """
        Calculate stiffness of the material given a material
        :param density: the density of the material
        :param penal: the interpolation power coefficient of the material
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


def get_load(shape: typing.Tuple[int, int, int]):
    """
    Defines the locations and forces applied to the structure
    :param shape: the number of nodes in the x, y, and z direction
    :return: flat array of forces with len of num_dof with format [Fx of n1, Fy of n2, Fz of n1, ... Fz of n]
    """
    x_nodes, y_nodes, z_nodes = shape
    # USER-DEFINED LOAD DOFs - where the forces are (top)
    il, jl, kl = np.meshgrid(range(0, x_nodes), y_nodes, range(0, z_nodes))  # Coordinates - this is a vector of forces at the end x distributed along 0 to max z
    loadnid = kl * (x_nodes + 1) * (y_nodes + 1) + il * (y_nodes + 1) + (y_nodes + 1 - jl)  # Node IDs
    load_dof = 3 * loadnid - 1  # DOFs
    num_dofs = 3 * (x_nodes + 1) * (y_nodes + 1) * (z_nodes + 1)  # number of degrees of freedom
    forces = np.zeros((num_dofs, 1))  # setup force
    forces[load_dof] = -1  # apply negative force at load
    return forces


def get_fixed(shape: typing.Tuple[int, int, int]):
    """
    Get the indices of which DOF are fixed in place
    :param shape: the number of nodes in the x, y, and z direction
    :return: the indices of the dof array that are fixed
    """
    x_nodes, y_nodes, z_nodes = shape
    # USER-DEFINED SUPPORT FIXED DOFs
    iif, jf, kf = np.meshgrid(range(0, x_nodes), 0, range(0, z_nodes))  # Coordinates - the supports are x=0 along the surface by iterating thorugh y&z
    fixednid = kf * (x_nodes + 1) * (y_nodes + 1) + iif * (y_nodes + 1) + (y_nodes + 1 - jf)  # Node IDs
    fixed_dof = np.array([3 * fixednid, 3 * fixednid - 1, 3 * fixednid - 2])  # DOFs
    return fixed_dof


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
        self.H, self.Hs = self._prep_filter(rmin)  # fixme: add better names

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.H*(x.flatten()/self.Hs.flatten()).T).reshape(x.shape)

    def _prep_filter(self, filter_radius: float) -> typing.Tuple[np.ndarray, np.ndarray]:
        x_nodes, y_nodes, z_nodes = self.size
        total_nodes = x_nodes * y_nodes * z_nodes
        ceil_r = math.ceil(filter_radius)
        # PREPARE FILTER - something to do with ill-posedness
        rough_len = total_nodes * (2*ceil_r)**3
        iH = np.ones(rough_len)#total_nodes * (2 * (ceil_r - 1) + 1) ** 2)
        jH = np.ones(iH.shape)
        sH = np.zeros(iH.shape)  # weight factors
        k = 0
        # iterates through nodes
        for k1 in range(0, z_nodes):
            for i1 in range(0, x_nodes):
                for j1 in range(0, y_nodes):
                    e1 = k1 * x_nodes * y_nodes + i1 * y_nodes + j1  # 1d index
                    # iterates through neighbors
                    for k2 in range(max(k1 - ceil_r, 0), min(k1 + (ceil_r), z_nodes)):
                        for i2 in range(max(i1 - ceil_r, 0), min(i1 + (ceil_r), x_nodes)):
                            for j2 in range(max(j1 - ceil_r, 0), min(j1 + (ceil_r), y_nodes)):
                                e2 = k2 * x_nodes * y_nodes + i2 * y_nodes + j2
                                iH[k] = e1
                                jH[k] = e2
                                sH[k] = max(0, filter_radius - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2 + (k1 - k2) ** 2))  # neighbor - distance = weight factor
                                k = k + 1
        H = coo_matrix((sH, (iH, jH)), shape=(total_nodes, total_nodes)).tocsc()  # each row is coordinate, each item in row is the corresponding weight
        Hs = H.sum(1)  # this is either the vertical or horizontal sum
        return H, Hs


class FEA:
    """
    Class the performs the analysis of how the structure deforms
    """
    def __init__(self, shape: typing.Tuple[int, int, int], free_dofs: np.ndarray):
        """
        Initialize the setup for the
        :param shape:
        :param free_dofs:
        """
        self.free_dofs = free_dofs
        x_nodes, y_nodes, z_nodes = shape
        total_nodes = x_nodes * y_nodes * z_nodes
        node_grd = np.array(range(0, (y_nodes + 1) * (x_nodes + 1))).reshape(
            (y_nodes + 1, x_nodes + 1))  # todo: think about deleteing thi
        node_ids = node_grd[:-1, :-1].reshape((y_nodes * x_nodes, 1))
        nodeidz = np.array(range(0, (z_nodes - 1) * (y_nodes + 1) * (x_nodes + 1) + 1, (y_nodes + 1) * (x_nodes + 1)))
        node_ids = np.tile(node_ids, (1, len(nodeidz))) + nodeidz
        element_dof_vec = 3 * (node_ids + 1).T.flatten()  # the first node of each cube
        relative_element = np.array([0, 1, 2, *(np.array([3, 4, 5, 0, 1, 2]) + 3 * y_nodes), -3, -2, -1, *(
                3 * (y_nodes + 1) * (x_nodes + 1) + np.array(
            [0, 1, 2, *(np.array([3, 4, 5, 0, 1, 2]) + 3 * y_nodes), -3, -2, -1]))])

        self.element_dof_mat = np.tile(element_dof_vec, (24, 1)).T + relative_element  # indices of the dof for each element
        # todo: figure out the variables below
        self.iK = np.kron(self.element_dof_mat.flatten(), np.ones((24, 1))).reshape((24 * 24 * total_nodes))  # rows of DOFs - used in creating global stiffness
        self.jK = np.kron(self.element_dof_mat.flatten(), np.ones((1, 24))).reshape((24 * 24 * total_nodes))  # cols of DOFs

    def solve(self, density: np.ndarray, material: Material, forces: np.ndarray):
        (x_nodes, y_nodes, z_nodes) = density.shape
        num_dofs = 3 * (x_nodes + 1) * (y_nodes + 1) * (z_nodes + 1)  # number of degrees of freedom
        total_nodes = x_nodes * y_nodes * z_nodes
        elementwise_stiffness = (material.element_stiffness.flatten() * np.repeat(material.stiffness(density), 576).reshape((total_nodes, 576))) .reshape((24 * 24 * total_nodes, 1))  # figure 15
        print(elementwise_stiffness[0:10])
        # global_stiffness = coo_matrix((elementwise_stiffness, (self.iK, self.jK)), shape=(num_dofs, num_dofs))#.tocsc()
        global_stiffness = coo_matrix((elementwise_stiffness, (self.iK, self.jK)), shape=(num_dofs, num_dofs), dtype=np.float64)
        global_stiffness = (global_stiffness + global_stiffness.T) / 2  # global stiffness matrix - averaging to persevere some past info (see python implement BESO)
        for row in global_stiffness.todense()[0:100]:
            print(row.tolist())
        quit()
        # todo: look into the iterative solver for this
        # solutions to: f = k * u where f = forces, u = displacements, k = stiffness
        free_stiff = global_stiffness[self.free_dofs, :][:, self.free_dofs].todense()
        quit()
        free_force = forces[self.free_dofs].reshape(len(self.free_dofs), 1)
        sol, residuals, rank, s = np.linalg.lstsq(free_force, free_stiff, rcond=0)
        return sol.reshape(len(self.free_dofs), 1)


class MinimumCompliance:
    def __init__(self, shape: typing.Tuple[int, int, int]):
        self.dc = np.ones(shape)
        self.dv = np.ones(shape)

    def optimize(self, density: np.ndarray, displacements: np.ndarray, material: Material, element_dof_mat: np.ndarray, volfrac: float):
        _, _, c = self.sensitivity_analysis(density, displacements, material, element_dof_mat)
        xnew, change = self.optimality_criteria(density, self.dc, self.dv, volfrac)
        return xnew

    def sensitivity_analysis(self, density: np.ndarray, displacements: np.ndarray, material: Material, element_dof_mat: np.ndarray):
        """
            Analyze the compliance of the sturcture and find the gradients of the material to minimize compliance while meeting the volume constraint
            :param density: The distribution of the material in the structure
            :param displacements: How much each node is displaced in the x, y, z material
            :param element_dof_mat: The indices of all the each node in the elements (row = element, cols = node indices)
            :param material: The material properties of the element
            :return: compliance gradient, volume gradient, compliance score
            """
        x_nodes, y_nodes, z_nodes = density.shape
        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS - figure 17-28 (minimum compliance)
        # fixme: the line below can be made more efficient
        print(f"d={displacements[element_dof_mat]}")
        ce = (displacements[element_dof_mat] * material.element_stiffness * displacements[element_dof_mat]).sum(axis=1).sum(axis=1).reshape([y_nodes, x_nodes, z_nodes])  # equation 9 - is this the strain energy / work
        print(f'{ce=}')
        c = (np.multiply(material.stiffness(density), ce)).sum()  # obj measure - compliance energy? is this the strain energy / work
        self.dc = -material.penal*(material.max_stiff-material.min_stiff) * density ** (material.penal-1) * ce  #-material.gradient(density) * ce
        print(-material.gradient(density))
        self.dv = np.ones(density.shape)  # this can probably just be calculated once
        print(f'{self.dc=}')
        # FILTERING AND MODIFICATION OF SENSITIVITIES - i think this is density filtering (see figure 3)
        self.dc[:] = Filter.smoothen(self.dc)  # derivative of the volume constraint - figure 21
        self.dv[:] = Filter.smoothen(self.dv)  # derivative of compliance -
        print(f'{self.dc=}')
        return self.dc, self.dv, c

    def sequential_quadratic_programming(self): pass

    def method_of_moving_asymptotes(self): pass

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
        x_nodes, y_nodes, z_nodes = x.shape
        density = np.zeros(x.shape)
        # looks like this imposes the volume constraint
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # max and min constrict between 0 and 1, move also limits the change
            xnew[:] = np.maximum(0.0, np.maximum(x-move,np.minimum(1.0, np.minimum(x+move, x * np.sqrt(-dc/dv/lmid)))))
            density[:] = Filter.smoothen(x)
            if density.mean() > volfrac:  #
                l1 = lmid
            else:
                l2 = lmid
        change = abs(xnew - x).max()
        return xnew, change


# === DISPLAY 3D TOPOLOGY (ISO-VIEW) === #
def display_3d(structure: np.ndarray):
    import matplotlib.pyplot as plt
    blocks = np.zeros(structure.shape, dtype=bool)
    blocks[structure>0.1] = True
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(np.where(structure > 0.1))#, edgecolor='k')
    plt.show()


# === main driver of the topology optimization === #
def top3d(x_nodes: int, y_nodes: int, z_nodes: int, volfrac: float, penal: float, rmin: float):
    """
    The main topology optimization function
    :param x_nodes: the x length
    :param y_nodes: the y length
    :param z_nodes: the x length
    :param volfrac: (1-porosity)
    :param penal: penalization power - reduce grey-scale values
    :param rmin: filter size
    :return: None
    """

    # can updat to pass a scenario, a material, and a filter
    mesh_shape = (x_nodes, y_nodes, z_nodes)
    num_dofs = 3 * (x_nodes + 1) * (y_nodes + 1) * (z_nodes + 1)  # number of degrees of freedom
    # USER-DEFINED parameters
    material = Material(poisson=0.3, max_stiff=1.0, min_stiff=1e-9, penal=penal)
    forces = get_load(mesh_shape)
    fixed_dof = get_fixed(mesh_shape)

    # PREPARE FINITE ELEMENT ANALYSIS
    displacements = np.zeros((num_dofs, 1))  # displacement, [dx1, dy1, dz1,... dz(8*n)]
    free_dofs = np.setdiff1d(range(0, num_dofs), fixed_dof)  # indices of the dof

    model = FEA(mesh_shape, free_dofs)

    smoother = Filter(rmin, mesh_shape)  # todo: fix so this looks used

    # INITIALIZE ITERATION
    x = volfrac * np.ones(mesh_shape)  # the start shape - even field of material
    density = x.copy()  # the densities
    change = 1
    opt = MinimumCompliance(mesh_shape)

    # START ITERATION
    for loop_i in range(200):
        if change < 0.01: break  # termination condition

        # analyze how the force causes displacement
        displacements[free_dofs, :] = model.solve(density, material, forces)
        # analyze the gradient of how to improve
        dc, dv, c = opt.sensitivity_analysis(density, displacements, material, model.element_dof_mat)
        # adjust material distribution
        x[:], change = opt.optimality_criteria(x, dc, dv, volfrac)
        density[:] = smoother.smoothen(x)
        # Update output
        print(f' Iteration: {loop_i} Objective:{c} Volume:{density.mean()} change:{change}')
    # display_3d(x)


def main2(x_nodes: int, y_nodes: int, z_nodes: int, volfrac: float, penal: float, rmin: float):
    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0
    # dofs:
    ndof = 3 * (x_nodes + 1) * (y_nodes + 1) * (z_nodes + 1)  # number of degrees of freedom
    shape = (x_nodes, y_nodes, z_nodes)
    total_nodes = x_nodes * y_nodes * z_nodes
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(total_nodes, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    dc = np.zeros((x_nodes, y_nodes, z_nodes), dtype=float)
    # FE: Build the index vectors for the for coo matrix format.
    KE = Material().element_stiffness
    mod = FEA(shape, None)
    edofMat = mod.element_dof_mat
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((24, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 24))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    f = Filter(rmin, shape)
    H, Hs = f.H, f.Hs
    # BC's and support
    dofs = np.arange(ndof)
    f = get_load(shape)
    fixed_dof = get_fixed(shape)
    free = np.setdiff1d(dofs, fixed_dof)

    u = np.zeros((ndof, 1))

    # Set loop counter and gradient vectors
    loop = 0
    g = 0
    change = 1
    dv = np.ones(total_nodes)
    dc = np.ones(total_nodes)
    ce = np.ones(total_nodes)
    while change > 0.01 and loop < 2000:
        loop = loop + 1
        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free, :][:, free]
        # Solve system
        u[free, 0] = spsolve(K, f[free, 0])
        # Objective and sensitivity
        ce[:] = (np.dot(u[edofMat].reshape(total_nodes, 24), KE) * u[edofMat].reshape(total_nodes, 24)).sum(1)
        obj = ((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()
        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        dv[:] = np.ones(total_nodes)
        # Density Filtering
        dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
        dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]
        # Optimality criteria
        xold[:] = x
        (x[:], g) = oc(x_nodes, y_nodes, z_nodes, x, volfrac, dc, dv, g)
        # Filter design variables
        xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
        # Compute the change by the inf. norm
        change = np.linalg.norm(x.reshape(total_nodes, 1) - xold.reshape(total_nodes, 1), np.inf)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format( \
            loop, obj, (g + volfrac * x_nodes * y_nodes) / (x_nodes * y_nodes), change))



def oc(x_nodes,y_nodes, z_nodes,x,volfrac,dc,dv, g):
	l1=0
	l2=1e9
	move=0.2
	# reshape to perform vector operations
	xnew=np.zeros(x_nodes*y_nodes*z_nodes)
	while (l2-l1)/(l1+l2)>1e-3:
		lmid=0.5*(l2+l1)
		xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
		gt=g+np.sum((dv*(xnew-x)))
		if gt>0 :
			l1=lmid
		else:
			l2=lmid
	return (xnew,gt)


if __name__ == '__main__':
    main2(5, 5, 5, 0.3, 3, 1.5)
    #top3d(5, 5, 5, 0.3, 3, 1.5)

# === ----------------------------------------------------------------- ===
# === The code is intended for educational purposes, and the details    ===
# === and extensions can be found in the paper:                         ===
# === K. Liu and A. Tovar, "An efficient 3D topology optimization code  ===
# === written in Matlab", Struct Multidisc Optim, 50(6): 1175-1196, 2014, =
# === doi:10.1007/s00158-014-1107-x                                     ===
# === ----------------------------------------------------------------- ===
