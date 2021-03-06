import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from sklearn.model_selection import KFold
from random import shuffle

import time
import tensorflow as tf

from neural_ode.ODESolvers import HeunsMethod


#
class NeuralODE:

    def __init__(self, model, n_dynamic,
                 n_external=int(0), n_ref=int(2),
                 loss_func=tf.keras.losses.MeanSquaredError(), solver=HeunsMethod()):
        self.model = model  # model, should change
        self.n_var = int(sum([tf.size(var) for var in model.variables]))  # number of variables
        self.loss_func = loss_func  # loss funcs, later
        self.solver = solver  # solver for integrating ODE
        self.n_ref = n_ref  # refinement of step_size
        self.n_dynamic = n_dynamic  # number of dynamic variables to model
        self.n_external = n_external  # number of external variables influencing dynamics

    @tf.function
    def grad_inps(self, x):
        #
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.model(x)
        J = tape.batch_jacobian(y, x)
        return J

    @tf.function
    def grad_params(self, x):
        #
        with tf.GradientTape() as tape:
            y = self.model(x)
        J_list = tape.jacobian(y, self.model.variables)
        return J_list

    def grad_inps_wrap(self, t, x, *args, batch=False):
        """
        Method to calculate jacobian of dynamical input - output
        :param t:
        :param x: dynamical input
        :param batch: if True, retain all batches and output will be [n_batch, ..., ...]
        if False, only first batch will remain and output will be [..., ...]
        :param args: list additional arguments,
        if there is external forcing then should have function for this forcing
        :return: jacobian for dynamical inputs only
        """
        # concatenate if have external forcing
        if self.n_external > 0:
            x_total = tf.concat([x, args[0](t)], axis=1)
        else:
            x_total = x
        J = self.grad_inps(x_total)
        # if dont need batches get rid of first dimension
        if batch:
            return J[:, :, 0:self.n_dynamic]
        else:
            return J[0, :, 0:self.n_dynamic]

    def grad_params_wrap(self, t, x, *args, batch=False):
        """
        Method to calculate jacobians of outputs to parameters
        And flatten them
        :param t:
        :param x:
        :param batch:
        :param args: list additional arguments,
        if there is external forcing then should have function for this forcing
        :return: jacobian for dynamical parameters
        """
        # concatenate if have external forcing
        if self.n_external > 0:
            x_total = tf.concat([x, args[0](t)], axis=1)
        else:
            x_total = x
        J_list = self.grad_params(x_total)
        npoints = tf.shape(x)[0]
        x_sh = self.n_dynamic
        J_flatten = [tf.reshape(item, (npoints, x_sh, tf.size(item) // (x_sh * npoints))) for item in J_list]
        J_flatten = tf.concat(J_flatten, axis=2)
        # if dont need batches get rid of first dimension
        if batch:
            return J_flatten
        else:
            return J_flatten[0, :, :]

    def unflatten_param(self, p) -> list:
        """
        Method to convert vector of parameters
        to list with tensors of size of variables
        :param p:
        :return:
        """
        list_p = []
        ix = 0
        for var in self.model.variables:
            n = tf.size(var)
            if len(tf.shape(p)) > 1:
                list_p.append(tf.reshape(tf.constant(p[0, ix:ix + n]), tf.shape(var)))
            else:
                list_p.append(tf.reshape(tf.constant(p[ix:ix + n]), tf.shape(var)))
            ix += n
        return list_p

    def ode_wrap(self, t, x, *args):
        """
        Wrapper for ode solver
        :param t: time. in our case just a placeholder
        :param x: dynamical inputs
        :param args: list additional arguments,
        if there is external forcing then should have function for this forcing
        :return: Value of derivatives according to model
        """
        if self.n_external > 0:
            external_arg = args[0](t)
            return self.model(tf.concat([x, external_arg], axis=1))
        else:
            return self.model(x)

    def pretrain(self, t_scale=5.0, step_size=0.1, n_epoch=10, external_scale=0.0,
                 opt=tf.keras.optimizers.Adam(learning_rate=0.05)):
        """
        Method to pretrain model so it's solution is decaying over time
        :param t_scale:
        :param step_size:
        :param n_epoch:
        :param external_scale:
        :param opt:
        :return:
        """
        t_span = tf.constant([0.0, t_scale], dtype=tf.float64)
        y0 = tf.ones((1, self.n_dynamic), dtype=tf.float64) * 0.1
        kwargs = {}
        if self.n_external > 0:
            x_external_interp = \
                lambda t: tf.ones((1, self.n_external), dtype=tf.float64) * external_scale
            kwargs['x_external'] = x_external_interp
        #
        if self.solver.requires_jacobian:
            kwargs['jac'] = self.grad_inps_wrap
        for i in range(n_epoch):
            with tf.GradientTape() as tape:
                sol, _ = self.solver(self.ode_wrap, t_span, y0,
                                     step_size=step_size, **kwargs)
                y_pred = sol['y'][-1, :]
                loss = self.loss_func(tf.zeros((1, self.n_dynamic), dtype=tf.float64), y_pred)
            dL_dp = tape.gradient(loss, self.model.trainable_variables)
            opt.apply_gradients(zip(dL_dp, self.model.trainable_variables))

    def forward_solve(self, t_eval, y0, **kwargs):
        """
        Method to solve model
        :param t_eval: Time values for data
        :param y0: Data to fit
        :param kwargs:
        :return:
        """
        if len(tf.shape(y0)) < 2:
            y0 = tf.expand_dims(y0, axis=0)
        step_size = (t_eval[1] - t_eval[0]) / self.n_ref
        t_span = tf.concat((t_eval[0], t_eval[-1]), axis=0)
        #
        if self.solver.requires_jacobian:
            kwargs['jac'] = self.grad_inps_wrap
        sol, _ = self.solver(self.ode_wrap, t_span, y0, step_size=step_size, **kwargs)
        return sol

    def aug_dyn(self, t, z_adj, sol_interp, *args):
        """
        Augmented dynamic function for backward integration
        :param t: time
        :param z_adj: adjoint vector state
        :param sol_interp: interpolation of solution to calculate derivatives of adjoint state
        :param args: additional argument should include external forcing function if needed
        :return: derivative of adjoint state
        """
        y_interp = sol_interp(t)
        df_dy = self.grad_inps_wrap(t, y_interp, *args)
        df_dp = self.grad_params_wrap(t, y_interp, *args)
        z_adj0 = z_adj[:, 0:self.n_dynamic]
        dz_adj = tf.concat((-tf.matmul(z_adj0, df_dy), -tf.matmul(z_adj0, df_dp)), axis=1)
        return dz_adj

    def backward_solve(self, t_eval, dL_dy, aug_dyn, **kwargs):
        """
        Backward solution for adjoint method
        :param t_eval: Time values for data
        :param dL_dy: derivatives of loss over output values of model
        :param aug_dyn: function for augmented dynamics
        :return:
        """
        y_adj0 = tf.concat((dL_dy[-1, :], tf.zeros([self.n_var], dtype=tf.float64)), axis=0)
        y_adj0 = tf.Variable(tf.expand_dims(y_adj0, axis=0))
        # t_print = time.time()
        for i in range(len(t_eval) - 1):
            step_size = (t_eval[-1 - i] - t_eval[-2 - i]) / self.n_ref
            t_span = tf.concat((t_eval[-1 - i], t_eval[-2 - i]), axis=0)
            sol_b, _ = self.solver(aug_dyn, t_span, y_adj0, step_size=step_size, **kwargs)
            # update
            y_adj0.assign(tf.expand_dims(sol_b['y'][-1, :], axis=0))
            y_adj0[:, 0:self.n_dynamic].assign(y_adj0[:, 0:self.n_dynamic] + dL_dy[-2 - i, :])
        # print(f'Solved backward: {time.time()-t_print}')
        return y_adj0

    def adjoint_method(self, t_eval, y_target, **kwargs):
        """
        Method to get gradients using adjoint method
        :param t_eval: Time values for data
        :param y_target: Data to fit
        :param kwargs:
        :return:
        """
        sol = self.forward_solve(t_eval, y_target[0, :], **kwargs)
        y_pred = sol['y'][::self.n_ref, :]
        # loss
        with tf.GradientTape() as tape:
            tape.watch(y_pred)
            loss = self.loss_func(y_target, y_pred)
        dL_dy = tape.gradient(loss, y_pred)
        #
        sol_np = interp1d(sol['t'].numpy(), sol['y'].numpy(), kind='linear', axis=0)
        # input for interpolation function should be zero size tensor
        sol_interp = lambda t: tf.expand_dims(tf.constant(sol_np(t)), axis=0)
        aug_dyn_fun = lambda t, z, *args: self.aug_dyn(t, z, sol_interp, *args)
        a = self.backward_solve(t_eval, dL_dy, aug_dyn_fun, **kwargs)
        return loss, dL_dy, a

    def usual_method(self, t_eval, y_target, **kwargs):
        """
        Method to get gradients using direct automatic differentiation
        :param t_eval: Time values for data
        :param y_target: Data to fit
        :param kwargs: missed_initial - tensors of missed initial conditions
            adjust_initial - boolean whether to calculate gradients for them,
            x_external - function for evaluating external forcing
        :return:
        """
        #
        if 'adjust_initial' in kwargs.keys():
            adjust_initial = kwargs['adjust_initial']
        else:
            adjust_initial = False
        with tf.GradientTape(persistent=adjust_initial) as tape:
            if 'missed_initial' in kwargs.keys():
                missed_initial = kwargs['missed_initial']
                if adjust_initial:
                    tape.watch(missed_initial)
                y0 = tf.concat([y_target[0, :], missed_initial], axis=0)
                n_tar = tf.size(y_target[0, :])
            else:
                y0 = y_target[0, :]
            sol = self.forward_solve(t_eval, y0, **kwargs)
            y_pred = sol['y'][::self.n_ref, :]
            if 'missed_initial' in kwargs.keys():
                y_pred = y_pred[:, 0:n_tar]
            loss = self.loss_func(y_target, y_pred)
        dL_dp = tape.gradient(loss, self.model.trainable_variables)
        grad_dict = {'dL_dp': dL_dp}
        # how to add gradient to initial condition
        if adjust_initial:
            dL_dy0 = tape.gradient(loss, missed_initial)
            grad_dict['dL_dy0'] = dL_dy0
        return loss, grad_dict

    def fit(self, t_eval, y_target, n_epoch=20, n_batch=5, adjoint_method=False,
            opt=tf.keras.optimizers.Adam(learning_rate=0.05), **kwargs) -> None:
        """
        Method to fit model to target data
        :param t_eval: Time values for data
        :param y_target: Data to fit
        :param n_epoch: Number of epoch
        :param n_batch: Number of batches (splitting) of data
        :param adjoint_method: boolean value whether to use adjoint method
        :param opt: optimizer to use
        :param kwargs:
            x_external - tensor of external forcing tag ,
            missing_derivative - list of tags, which derivatives are missing,
            adjust_initial - boolean whether to adjust initial missing condition or not
        :return: None
        """
        # collect kwargs for method
        kwargs_inp = {}
        if self.n_external > 0:
            if 'x_external' in kwargs.keys():
                x_external = kwargs['x_external']
                # create interpolation
                x_external_interp_np = interp1d(t_eval.numpy(),
                                                x_external.numpy(), kind='linear', axis=0)
                # input for interpolation function should be zero size tensor
                x_external_interp = lambda t: tf.expand_dims(tf.constant(x_external_interp_np(t)), axis=0)
                kwargs_inp = {'x_external': x_external_interp}
            else:
                print('No external data')
                return None
        #
        shuffle_batches = True
        # create batches
        if n_batch > 1:
            kf = KFold(n_splits=n_batch)
            ix_list = [ix_train for __, ix_train in kf.split(t_eval)]
        else:
            ix_list = [np.arange(0, len(t_eval))]
        # check whether there are missing derivative and initialise them
        mis_deriv = False
        adjust_initial = False
        if 'missing_derivative' in kwargs.keys():
            mis_deriv = True
            if 'adjust_initial' in kwargs.keys():
                kwargs_inp['adjust_initial'] = kwargs['adjust_initial']
            else:
                kwargs_inp['adjust_initial'] = False
            adjust_initial = kwargs_inp['adjust_initial']
            # initialise learning rate for init conditions
            if kwargs_inp['adjust_initial']:
                lr_init = tf.constant(0.1, dtype=tf.float64)
            # initialise list derivatives for each fold with the simplest formula
            dt = t_eval[1] - t_eval[0]
            miss_initial = []
            for deriv_ix in kwargs['missing_derivative']:
                for ix_train in ix_list:
                    deriv = (tf.gather(y_target[ix_train[1], :], indices=deriv_ix) -
                             tf.gather(y_target[ix_train[0], :], indices=deriv_ix)) / dt
                    if len(deriv.shape) < 1:
                        deriv = tf.expand_dims(deriv, axis=0)
                    miss_initial.append(tf.Variable(deriv))
        loss_list = []
        # start epochs
        t_tot = time.time()
        for i in range(n_epoch):
            print(f'--- Epoch #{i + 1} ---')
            t_epoch = time.time()
            epoch_loss = 0.0
            if shuffle_batches:
                # if there are missing init conditions should shuffle them together with data
                if mis_deriv:
                    to_shuffle = list(zip(ix_list, miss_initial))
                    shuffle(to_shuffle)
                    ix_list, miss_initial = zip(*to_shuffle)
                    ix_list = list(ix_list)
                    miss_initial = list(miss_initial)
                else:
                    shuffle(ix_list)
            # loop over every fold
            for ix, ix_train in enumerate(ix_list):
                # write missing init cond to kwargs
                if mis_deriv:
                    kwargs_inp['missed_initial'] = miss_initial[ix]
                if adjoint_method:
                    loss, dL_dy, a = self.adjoint_method(tf.gather(t_eval, indices=ix_train),
                                                         tf.gather(y_target, indices=ix_train),
                                                         **kwargs_inp)
                    grads_p = a[0, self.n_dynamic:]
                    grads_list = self.unflatten_param(grads_p)
                else:
                    loss, grads_dict = self.usual_method(tf.gather(t_eval, indices=ix_train),
                                                         tf.gather(y_target, indices=ix_train), **kwargs_inp)
                    grads_list = grads_dict['dL_dp']
                    if adjust_initial:
                        miss_initial[ix].assign_sub(grads_dict['dL_dy0'] * lr_init)
                epoch_loss += loss
                opt.apply_gradients(zip(grads_list, self.model.trainable_variables))
            loss_list.append(epoch_loss)
            print(f'Total loss of epoch: {epoch_loss}')
            print(f'Elapsed time for epoch: {time.time() - t_epoch}')
        #
        print(f'Total elapsed time: {time.time() - t_tot}')
        # figure with loss
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(loss_list)
        # build solution and compare
        # find missing initial condition for first fold
        if mis_deriv:
            init_cond = [miss_initial[ix] for ix, ix_train in enumerate(ix_list) if ix_train[0] == 0][0]
            y0 = tf.concat([y_target[0, :], init_cond], axis=0)
        else:
            y0 = y_target[0, :]
        sol2 = self.forward_solve(t_eval, y0, **kwargs_inp)
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(t_eval.numpy(), y_target[:, 0].numpy())
        ax.plot(sol2['t'].numpy(), sol2['y'][:, 0].numpy())
