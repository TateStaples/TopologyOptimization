__author__ = "Tate Staples"

import numpy as np
import nlopt
import time


class Parameter:
    """Class to wrap value and gradient for objective and constraint values"""
    def __init__(self, value, gradient, maximum=None): self.value = value; self.gradient = gradient; self.max = maximum


class Optimizer:
    """
    Implements the Method of Moving asymptotes optimizer algorithm to apply the calculated gradients and constraints
    """
    def __init__(self, shape, update, passive_elem=()):
        """
        Set up the optimization parameters
        :param shape: shape of the structure
        :param update: the code called periodically between each cycle
        :param passive_elem: where the contents should stay at min
        """
        x, y, z = self.shape = shape
        self.total_nodes = n = x * y * z  # number of design variables
        self.min_densities = 0.05 * np.ones(n)  # minimum values for design
        self.max_densities = np.ones(n)  # max values for design
        self.passive = passive_elem.flatten('F')
        self.min_densities[self.passive] = 1e-9; self.max_densities[self.passive] = 1e-9;
        self.opt = opt = nlopt.opt(nlopt.LD_MMA, n)
        self.updater = update
        self.results = None
        self.change = 1
        self.prev = np.ones(n)
        self.iteration = 0
        self.start_time = time.time()
        opt.set_lower_bounds(self.min_densities)
        opt.set_upper_bounds(self.max_densities)
        opt.set_maxeval(2000)
        opt.set_xtol_abs(0.01)
        # opt.set_ftol_rel(1e-4)

    def optimize(self, x, obj, *constraints) -> np.ndarray:
        """
        Starts the optimization loop. This function will run until final result.
        Any intermediate should be in the update function specified at initialization
        :param x: the initial structure design
        :param obj: The objective function. Should be of form(density, grad) -> score & update grad array
        :param constraints: Sequence of inequality constraints of form(density, grad) -> score & update grad array
        :return: The optimized structure
        """
        shape = x.shape
        x = x.flatten('F')
        x[:] = np.minimum(np.maximum(x, self.min_densities), self.max_densities)
        for c in constraints:
            self.opt.add_inequality_constraint(c)
        self.opt.set_min_objective(self._objective(obj))
        return self.opt.optimize(x).reshape(shape, order='F')

    def _objective(self, func):
        """
        Private method that adds a extra update when the optimization loop.
        Specifically in runs the next iteration of the optimization before each calculation
        :param func: The function that return the info on the objective variable
        :return: # wrapped objective function
        """
        def obj(x, grad):
            self.change = abs(x - self.prev).max()
            print(f"itr:{self.iteration}\t∆x:{round(self.change*100, 1)}% ({round(time.time() - self.start_time, 2)} s)", end='\t')
            self.prev[:] = x.copy();
            self.iteration += 1;
            self.start_time = time.time()
            self.updater(x.reshape(self.shape, order='F'))
            return func(x, grad)
        return obj