__author__ = "Tate Staples"

import numpy as np

import gyroidizer
import utils
from materials import Gyroid
from load import LoadCase
from fea import FEA
from IO import save, load, load_cache, Display
from sensitivity import stress_sensitivity, compliance_sensitivity
import sensitivity
from optimizer import Optimizer
import filters


def setup(shape, units_per_meter):
    """
    Set up some of the base details
    :param shape:
    :param units_per_meter: The number of gyroids per meter (used for standardizing units)
    """
    utils.units_per_meter = units_per_meter
    filters.prep_filter(shape, filter_radius=2.5)
    sensitivity.dv = filters.filter_sensitivities(np.ones(shape))
    sensitivity.dc = np.ones(shape)


def project(radius: int, height: int, units_per_meter=1, cache: str = None):
    """
    Main cylinder generation method
    :param radius: radius of the cylinder
    :param height: height of the cylinder
    :param units_per_meter: gyroids per meter
    :param cache: previous design path. Will change intial design to speed up generation
    :return:
    """
    shape = (height, radius*2, radius*2)  # the shape of our grid - used to create arrays
    setup(shape, units_per_meter)
    passive = utils.passive_cylinder(shape)
    # Allocate design variables (as array), initialize and allocate sens.
    volfrac = 0.24 * (np.pi/4)  # mult density times volume ratio of cylinder vs rect prism
    x = volfrac * np.ones(shape, dtype=float) if cache is None else load_cache(cache, radius, height)
    material = Gyroid(0.3, utils.scale_stress(119e9), 1e-19, 3)  # define the material properties of you structure
    load1 = LoadCase.compress(shape).add_force((0, 0, 0), utils.dof_passive(shape))
    # load2 = LoadCase.torsion(shape)
    # https://www.sciencedirect.com/science/article/pii/S1359645418306293
    modeler = FEA(shape, material, load1)  # physics simulations
    yield_stress = utils.scale_stress(40e6)  # base titanium yield in 40 MPa
    d = Display(shape)

    # Set loop counter and gradient vectors

    def update(density):
        print()
        density = density.reshape(shape, order='F')
        modeler.displace(density)

    def vol_update(density, grad):
        density = density.reshape(shape, order='F')
        if grad.size > 0: grad[:] = np.ones(grad.shape)#sensitivity.dv.flatten('F')
        mean = density.mean()
        print(f"Vol: {round(mean * 100, 1)}", end="\t")
        return mean - volfrac

    def stress_update(density, grad):
        density = density.reshape(shape, order='F')
        modeler.calc_stress(density)
        # d.display_3d(density, modeler.von_mises_stress.reshape(shape, order="F"))
        if grad.size > 0: grad[:] = stress_sensitivity(density, modeler).flatten('F')
        print(f"Stress: {round(utils.unscale_stress(modeler.max_stress), 2)} ({round(modeler.max_stress/yield_stress, 2)})", end="\t")
        return modeler.max_stress - yield_stress

    def compliance_update(density, grad):
        density = density.reshape(shape, order='F')
        modeler.calc_strain(density)
        if grad.size > 0: grad[:] = compliance_sensitivity(density, modeler).flatten('F')
        d.display_3d(density, grad.reshape(shape, order="F"))
        d.show()
        print(f"Comp:{round(modeler.compliance, 2)}", end="\t")
        return modeler.compliance

    opt = Optimizer(shape, update, passive)  # updates the structure to new distribution
    x[:] = np.maximum(opt.min_densities.reshape(x.shape, order='F'), x)
    try:
        x = opt.optimize(x, compliance_update, vol_update)
    except KeyboardInterrupt:
        save(x, "saved_progress")
        print("optimization interrupted by user request")
        quit()
    modeler.calc_stress(x)
    fname = f"gr{radius}h{height}" if isinstance(material, Gyroid) else f"r{radius}h{height}"
    save(x, fname)
    d.make_animation(fname)
    d.display_3d(x, modeler.von_mises_stress.reshape(shape, order="F"))
    d.show()
    d.save(fname)


def double(arr):
    """
    Convert half design to full design
    :param arr:
    :return: Full design
    """
    return np.concatenate((arr, np.flip(arr, axis=0)), axis=0)


def run_load():
    gyroidizer.gyroidize(load("gr10h80"), scale=1/4, resolution=25j)


# todo: add command line intialization option


if __name__ == '__main__':
    s = load("gr10h80")
    gyroidizer.gyroidize(s, resolution=25j, scale=1/4)
    s.mean()
    d = Display(s.shape)
    d.display_3d(s, np.ones(s.shape))
    d.show()
    quit()
    project(10, 80, cache="gr10h100", units_per_meter=1000/4)
