import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def Newtons_method(y0, resid, jac, **opts):
    opts_default = {'max_iter': int(20), 'f_tol': tf.constant(1e-6, dtype=tf.float64)}
    opts_default.update(**opts)
    finished = False
    iter = int(0)
    resid_curr = resid(y0)
    jac_curr = jac(y0)
    y1 = y0
    while not finished:
        dy = tf.linalg.solve(-tf.transpose(jac_curr), tf.transpose(resid_curr) )
        y1 = y1 + tf.transpose(dy)
        resid_curr = resid(y1)
        jac_curr = jac(y1)
        iter += 1
        if tf.norm(resid_curr) < opts_default['f_tol']:
            finished = True
        if iter > opts_default['max_iter']:
            finished = True
            print('Max iterations is reached')
    #print(tf.norm(resid_curr))
    return y1

class ODESolvertf:

    def __init__(self):
        self.is_implicit = False
        self.is_multistep = False

    def step_calculate(self, ode_fun, t, y, step_size, *add_args):
        return None, None

    def __call__(self, ode_fun, tspan, y0,  **kwargs):
        """
        Method to solve ODE Problem
        :param ode_fun: function to calculate derivatives.
            Should take time, [1xN] tensor, *args for external function
        :param tspan:
        :param y0:
        :param kwargs:
        :return:
        """
        dT = tspan[1] - tspan[0]
        if 'step_size' in kwargs.keys():
            n_step = int(tf.math.ceil(tf.abs(dT) / kwargs['step_size']))
            step_size = dT / n_step
        elif 'n_step' in kwargs.keys():
            n_step = int(kwargs['n_step'])
            step_size = dT / n_step
        if 'x_external' in kwargs.keys():
            add_args = [kwargs['x_external']]
        else:
            add_args = []
        if self.is_implicit and 'jac' not in kwargs.keys():
            print('Implicit solver needs specified jacobian')
            return None, None
        if self.is_implicit and 'jac' in kwargs.keys():
            kwargs2 = {'jac': kwargs['jac']}
        else:
            kwargs2 = {}
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
            y_next, err_estimation = self.step_calculate(ode_fun, sol_dict['t'][i:i+2], y_curr,
                                                         step_size, *add_args, **kwargs2)
            sol_list.append(y_next)
            err_list.append(tf.norm(err_estimation))
        err_list = tf.concat(err_list, axis=0)
        sol_dict['y'] = tf.concat(sol_list, axis=0)
        return sol_dict, err_list


class EulerMethod(ODESolvertf):

    def step_calculate(self, ode_fun, t, y, step_size, *add_args, **kwargs):
        y_pr = y + step_size * ode_fun(t[0], y, *add_args)
        err_estimation = tf.zeros(tf.shape(y))
        return y_pr, err_estimation


class HeunsMethod(ODESolvertf):

    def step_calculate(self, ode_fun, t, y, step_size, *add_args, **kwargs):
        y_pr = y + step_size * ode_fun(t[0], y, *add_args)
        y_corr = y + step_size / 2 * (ode_fun(t[0], y, *add_args) +
                                      ode_fun(t[1], y_pr, *add_args))
        err_estimation = y_corr - y_pr
        return y_corr, err_estimation


# TODO: check why backward Euler is wrong and check Newtons method
class BackwardEulerMethod(ODESolvertf):

    def __init__(self):
        self.is_implicit = True
        self.is_multistep = False

    def step_calculate(self, ode_fun, t, y, step_size, *add_args, **kwargs):
        y_pr = y + step_size * ode_fun(t[0], y, *add_args)
        resid = lambda x: x - y - step_size*ode_fun(t[1], x, *add_args)
        jac_solv = lambda x: -tf.eye(len(y), dtype=tf.float64) - step_size*kwargs['jac'](x)
        y_corr = Newtons_method(y_pr, resid, jac_solv)
        err_estimation = tf.zeros(tf.shape(y))
        return y_corr, err_estimation
