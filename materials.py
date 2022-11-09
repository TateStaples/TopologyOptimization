import numpy as np


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
        # return (-482.65 * np.power(density, 3) + 938.34 * np.power(density, 2) + 27.693 * density + self.min_stiff) * self.max_stiff / 3145e6
        # https://www.sciencedirect.com/science/article/pii/S1359645418306293
        return 0.293 * np.power(density, 2) * self.max_stiff + self.min_stiff

    def gradient(self, density: np.ndarray) -> np.ndarray:
        # return (3 * -482.65 * np.power(density, 2) + 2 * 938.34 * density + 27.693) * self.max_stiff / 3145e6  # ratio of titanium to PLA
        return 2 * 0.293 * density * self.max_stiff

