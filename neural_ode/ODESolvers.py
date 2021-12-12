import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ODESolvertf:

    def __init__(self):
        self.type_explicit = 'explicit'
        self.type_step = 'single_step'

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
                                                         step_size, *add_args)
            sol_list.append(y_next)
            err_list.append(tf.norm(err_estimation))
        err_list = tf.concat(err_list, axis=0)
        sol_dict['y'] = tf.concat(sol_list, axis=0)
        return sol_dict, err_list


class EulerMethod(ODESolvertf):

    def step_calculate(self, ode_fun, t, y, step_size, *add_args):
        y_pr = y + step_size * ode_fun(t[0], y, *add_args)
        err_estimation = tf.zeros(tf.shape(y))
        return y_pr, err_estimation


class HeunsMethod(ODESolvertf):

    def step_calculate(self, ode_fun, t, y, step_size, *add_args):
        y_pr = y + step_size * ode_fun(t[0], y, *add_args)
        y_corr = y + step_size / 2 * (ode_fun(t[0], y, *add_args) +
                                      ode_fun(t[1], y_pr, *add_args))
        err_estimation = y_corr - y_pr
        return y_corr, err_estimation
