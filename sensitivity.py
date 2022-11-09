import numpy as np

from fea import FEA, stress_relaxation, norm_aggregation
from filters import filter_sensitivities

dv = dc = None


def compliance_sensitivity(density: np.ndarray, model: FEA):
    """
    Calculate the gradient of compliance with respect to density (dCdX)
    :param density: densities of the structure
    :param model: the physical model of the structure. Contains important properties
    :return: Compliance gradient, Volume gradient, current compliance
    """
    # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS - figure 17-28 (minimum compliance)
    dc[:] = -model.material.gradient(
        density) * model.strain  # i dont actually know where this is derived in the paper
    # Density Filtering (prevents irregular gaps)
    dc[:] = filter_sensitivities(dc)
    return dc


def stress_sensitivity(density: np.ndarray, model: FEA):
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
    total_nodes = density.size
    strain_matrix, elastic_matrix = model.material.strain_matrix, model.material.elastic_matrix

    # derivative of von mies stress with respect to relaxed stress
    DvmDrs = np.zeros((total_nodes, 6))
    # derivative of p normal stress (obj) with respect to von mises stress
    DpnDvm = (model.von_mises_stress ** norm_aggregation).sum() ** (1 / norm_aggregation - 1)

    index_matrix = model.element_dof_mat.T
    for i in range(0, total_nodes):
        DvmDrs[i, 0] = 1 / 2 / model.von_mises_stress[i] * (
                    2 * model.stress[i, 0] - model.stress[i, 1] - model.stress[i, 2])
        DvmDrs[i, 1] = 1 / 2 / model.von_mises_stress[i] * (
                    2 * model.stress[i, 1] - model.stress[i, 0] - model.stress[i, 2])
        DvmDrs[i, 2] = 1 / 2 / model.von_mises_stress[i] * (
                    2 * model.stress[i, 2] - model.stress[i, 0] - model.stress[i, 1])
        DvmDrs[i, 3] = 3 / model.von_mises_stress[i] * model.stress[i, 3]
        DvmDrs[i, 4] = 3 / model.von_mises_stress[i] * model.stress[i, 4]
        DvmDrs[i, 5] = 3 / model.von_mises_stress[i] * model.stress[i, 5]

    # calculation of T1
    # sum(vm^(p-1) * dPNdVM[dVMds.T * dNdx * s]) - beta is between the square brackets - Eq. 20
    beta = np.zeros((total_nodes, 1))
    for i in range(0, total_nodes):
        element_displacement = model.displacement[model.element_dof_mat[i, :], :].T.reshape((24, 1), order="F")
        # Eq. 19 - https://link.springer.com/article/10.1007/s11081-021-09675-3#Equ19
        DnDx = stress_relaxation * (density.flatten('F')[i]) ** (stress_relaxation - 1)
        # Eq. 21  - https://link.springer.com/article/10.1007/s11081-021-09675-3#Equ21
        DsDx = elastic_matrix @ strain_matrix @ element_displacement
        beta[i] = DnDx * model.von_mises_stress[i] ** (norm_aggregation - 1) * DvmDrs[i, :] @ DsDx
    T1 = DpnDvm * beta

    # calculation of T2
    # 𝛾 [gama] = sum(n(x) * dPNdVM * dSdX.T * dVMdRS)=
    gama = np.zeros(model.displacement.shape)
    for i in range(0, total_nodes):
        index = index_matrix[:, i]
        DsDx = elastic_matrix @ strain_matrix
        n = density.flatten('F')[i] ** stress_relaxation
        update = n * DpnDvm * DsDx.T @ DvmDrs[i, :].T * model.von_mises_stress[i] ** (norm_aggregation - 1)
        gama[index] = gama[index] + update.reshape((24, 1))
    # K𝜆 [lambda] = 𝛾 * K.inv()
    lamda = np.zeros(model.displacement.shape)
    lamda[model.free_dofs, 0] = model.solve(model.global_stiffness[model.free_dofs, :][:, model.free_dofs], gama[model.free_dofs, :])

    T2 = np.zeros((total_nodes, 1))
    for i in range(0, total_nodes):
        index = index_matrix[:, i]
        # T2 = -𝜆.T * dKdx * U [Eq. 29]
        dKdX = model.material.gradient(density).flatten('F')[i] * model.material.element_stiffness
        T2[i] = -lamda[index].T @ dKdX @ model.displacement[index]

    DpnDx = T1 + T2
    return filter_sensitivities(DpnDx.reshape(density.shape, order="F"))