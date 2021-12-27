import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def newtons_method(y0, resid, jac, **opts):
    """
    Function for implementing newtons method for solving nonlinear equations
    :param y0: shape [1, n]
    :param resid: shape [1, n]
    :param jac: shape [n, n]
    :param opts:
    :return:
    """
    opts_default = {'max_iter': int(20), 'f_tol': tf.constant(1e-6, dtype=tf.float64),
                    'is_vec_row': True}
    opts_default.update(**opts)
    finished = False
    iter = int(0)
    resid_curr = tf.transpose(resid(y0))
    jac_curr = jac(y0)
    y1 = y0
    while not finished:
        dy = tf.linalg.solve(jac_curr, -resid_curr)
        y1 = y1 + tf.transpose(dy)
        resid_curr = tf.transpose(resid(y1))
        jac_curr = tf.transpose(jac(y1))
        iter += 1
        if tf.norm(resid_curr) < opts_default['f_tol']:
            finished = True
        if iter > opts_default['max_iter']:
            finished = True
            print('Max iterations is reached')
    return y1


class ODESolvertf:

    def __init__(self):
        self.requires_jacobian = False
        self.is_multistep = False

    def step_calculate(self, ode_fun, t, y, step_size, *args_step, **kwargs_step):
        """

        :param ode_fun: function for derivatives
        :param t: current time and next time
        :param y: current value of unknown variable
        :param step_size:
        :param args_step:
        :param kwargs_step:
        :return:
        """
        return None, None

    def __call__(self, ode_fun, tspan, y0,  **kwargs):
        """
        Method to solve ODE Problem
        :param ode_fun: function to calculate derivatives.
            Should take time, [1xN] tensor, *args for external function
        :param tspan:
        :param y0:
        :param kwargs: n_step or step_size,
        x_external - function for evaluating external forcing,
        jac - jacobian
        :return:
        """
        # find step
        dT = tspan[1] - tspan[0]
        if 'step_size' in kwargs.keys():
            n_step = int(tf.math.ceil(tf.abs(dT) / kwargs['step_size']))
            step_size = dT / n_step
        elif 'n_step' in kwargs.keys():
            n_step = int(kwargs['n_step'])
            step_size = dT / n_step
        #
        kwargs_step = {}
        args_step = []
        if 'x_external' in kwargs.keys():
            args_step.append(kwargs['x_external'])
        if self.requires_jacobian and 'jac' not in kwargs.keys():
            print('Implicit solver needs specified jacobian')
            return None, None
        if self.requires_jacobian and 'jac' in kwargs.keys():
            kwargs_step['jac'] = kwargs['jac']
        N_p = n_step + int(1)
        sol_dict = {}
        sol_dict['t'] = tf.linspace(tspan[0], tspan[1], N_p)
        #
        sol_list = []
        sol_list.append(y0)
        err_list = []
        # sol_list - list of tensors
        for i in range(n_step):
            # tensor shape should be [1, N]
            y_curr = sol_list[i]
            args = []
            y_next, err_estimation = self.step_calculate(ode_fun, sol_dict['t'][i:i+2], y_curr,
                                                         step_size, *args_step, **kwargs_step)
            sol_list.append(y_next)
            err_list.append(tf.norm(err_estimation))
        err_list = tf.concat(err_list, axis=0)
        sol_dict['y'] = tf.concat(sol_list, axis=0)
        return sol_dict, err_list


class EulerMethod(ODESolvertf):

    def step_calculate(self, ode_fun, t, y, step_size, *args, **kwargs):
        y_pr = y + step_size * ode_fun(t[0], y, *args)
        err_estimation = tf.zeros(tf.shape(y))
        return y_pr, err_estimation


class HeunsMethod(ODESolvertf):

    def step_calculate(self, ode_fun, t, y, step_size, *args, **kwargs):
        fi = ode_fun(t[0], y, *args)
        y_pr = y + step_size * fi
        y_corr = y + step_size / 2 * (fi + ode_fun(t[1], y_pr, *args))
        err_estimation = y_corr - y_pr
        return y_corr, err_estimation


class BackwardEulerMethod(ODESolvertf):

    def __init__(self):
        self.requires_jacobian = True
        self.is_multistep = False

    def step_calculate(self, ode_fun, t, y, step_size, *args, **kwargs):
        y_pr = y + step_size * ode_fun(t[0], y, *args)
        resid = lambda x: y + step_size*ode_fun(t[1], x, *args) - x
        jac_solv = lambda x: step_size*kwargs['jac'](t[1], x, *args) - tf.eye(tf.size(y), dtype=tf.float64)
        y_corr = newtons_method(y_pr, resid, jac_solv)
        err_estimation = tf.zeros(tf.shape(y))
        return y_corr, err_estimation


class TrapezoidRuleMethod(ODESolvertf):

    def __init__(self):
        self.requires_jacobian = True
        self.is_multistep = False

    def step_calculate(self, ode_fun, t, y, step_size, *args, **kwargs):
        fi = ode_fun(t[0], y, *args)
        y_pr = y + step_size * fi
        resid = lambda x: y + step_size/2 * (fi + ode_fun(t[1], x, *args)) - x
        jac_solv = lambda x: step_size/2*kwargs['jac'](t[1], x, *args) - tf.eye(tf.size(y), dtype=tf.float64)
        y_corr = newtons_method(y_pr, resid, jac_solv)
        err_estimation = tf.zeros(tf.shape(y))
        return y_corr, err_estimation


# Taylor series expansion. have to check
class TaylorSecondOrder(ODESolvertf):

    def __init__(self):
        self.requires_jacobian = True
        self.is_multistep = False

    def step_calculate(self, ode_fun, t, y, step_size, *args, **kwargs):
        fi = ode_fun(t[0], y, *args)
        jac0 = kwargs['jac'](t[0], y, *args)
        y_pr = y + step_size * fi + step_size**2/2 * tf.transpose(tf.matmul(jac0, tf.transpose(fi)) )
        err_estimation = tf.zeros(tf.shape(y))
        return y_pr, err_estimation
