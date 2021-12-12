import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from sklearn.model_selection import KFold
from random import shuffle

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from neural_ode.ODESolvers import HeunsMethod


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
        """
        Method to calculate jacobians of dynamical input - output
        :param x:
        :return:
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.model(x)
        J = tape.batch_jacobian(y, x)
        return J[:, :, 0:self.n_dynamic]

    @tf.function
    def grad_params(self, x):
        """
        Method to calculate jacobians of outputs to parameters
        And flatten them
        :param x:
        :return:
        """
        with tf.GradientTape() as tape:
            y = self.model(x)
        npoints = tf.shape(x)[0]
        x_sh = self.n_dynamic
        J_list = tape.jacobian(y, self.model.variables)
        J_flatten = [tf.reshape(item, (npoints, x_sh, tf.size(item) // (x_sh * npoints))) for item in J_list]
        J_flatten = tf.concat(J_flatten, axis=2)
        return J_flatten

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
                print(p)
                list_p.append(tf.reshape(tf.constant(p[0, ix:ix + n]), tf.shape(var)))
            else:
                list_p.append(tf.reshape(tf.constant(p[ix:ix + n]), tf.shape(var)))
            ix += n
        return list_p

    def ode_wrap(self, t, x, *args):
        """
        Wrapper for ode solver
        :param t:
        :param x:
        :param args:
        :return:
        """
        if self.n_external > 0:
            external_arg = args[0](t)
            return self.model(tf.concat([x, external_arg], axis=1))
        else:
            return self.model(x)

    def pretrain(self, t_scale=5.0, step_size=0.1, n_epoch=10, external_scale = 0.1,
                 opt=tf.keras.optimizers.Adam(learning_rate=0.05)):
        """
        Method to pretrain model so it's solution is decaying over time
        :param t_scale:
        :param step_size:
        :param n_epoch:
        :param opt:
        :return:
        """
        t_span = tf.constant([0.0, t_scale], dtype=tf.float64)
        y0 = tf.ones((1, self.n_dynamic), dtype=tf.float64) * 0.1
        if self.n_external > 0:
            x_external_interp = \
                lambda t: tf.ones((1, self.n_external), dtype=tf.float64) * external_scale
            dict_external = {'x_external': x_external_interp}
        else:
            dict_external = {}
        for i in range(n_epoch):
            with tf.GradientTape() as tape:
                sol, _ = self.solver(self.ode_wrap, t_span, y0,
                                     step_size=step_size, **dict_external)
                y_pred = sol['y'][-1, :]
                loss = self.loss_func(tf.zeros((1, self.n_dynamic), dtype=tf.float64), y_pred)
            dL_dp = tape.gradient(loss, self.model.trainable_variables)
            opt.apply_gradients(zip(dL_dp, self.model.trainable_variables))

    def forward_solve(self, t_eval, y0, **kwargs):
        """
        Method to solve model
        :param t_eval:
        :param y0:
        :param kwargs:
        :return:
        """
        if len(tf.shape(y0)) < 2:
            y0 = tf.expand_dims(y0, axis=0)
        step_size = (t_eval[1] - t_eval[0]) / self.n_ref
        t_span = tf.concat((t_eval[0], t_eval[-1]), axis=0)
        sol, _ = self.solver(self.ode_wrap, t_span, y0, step_size=step_size, **kwargs)
        return sol

    def construct_aug_dyn(self, sol, **kwargs):
        """
        Method to construct augmented dynamics function for adjoint method
        :param sol:
        :param kwargs:
        :return:
        """
        # make interpolation of sol
        sol_np = interp1d(sol['t'].numpy(), sol['y'].numpy(), kind='linear', axis=0)
        sol_interp = lambda t: tf.expand_dims(tf.constant(sol_np(t)), axis=0)

        def func(t, z_adj):
            y_interp = sol_interp(t)
            if 'x_external' in kwargs.keys():
                x_ext = kwargs['x_external'](t)
                y_total = tf.concat([y_interp, x_ext], axis=1)
                df_dy = self.grad_inps(y_total)
                df_dp = self.grad_params(y_total)
            else:
                df_dy = self.grad_inps(y_interp)
                df_dp = self.grad_params(y_interp)
            # get rid of first dimension
            df_dy = df_dy[0, :, :]
            df_dp = df_dp[0, :, :]
            z_adj0 = z_adj[:, 0:self.n_dynamic]
            dz_adj = tf.concat((-tf.matmul(z_adj0, df_dy), -tf.matmul(z_adj0, df_dp)), axis=1)
            return dz_adj
        return func

    def backward_solve(self, t_eval, dL_dy, aug_dyn):
        """
        Backward solution for adjoint method
        :param t_eval:
        :param dL_dy:
        :param aug_dyn:
        :return:
        """
        y_adj0 = tf.concat((dL_dy[-1, :], tf.zeros([self.n_var], dtype=tf.float64)), axis=0)
        y_adj0 = tf.Variable(tf.expand_dims(y_adj0, axis=0))
        # t_print = time.time()
        for i in range(len(t_eval) - 1):
            step_size = (t_eval[-1 - i] - t_eval[-2 - i]) / self.n_ref
            t_span = tf.concat((t_eval[-1 - i], t_eval[-2 - i]), axis=0)
            sol_b, _ = self.solver(aug_dyn, t_span, y_adj0, step_size=step_size)
            # update
            y_adj0.assign(tf.expand_dims(sol_b['y'][-1, :], axis=0))
            y_adj0[:, 0:self.n_dynamic].assign(y_adj0[:, 0:self.n_dynamic] + dL_dy[-2 - i, :])
        # print(f'Solved backward: {time.time()-t_print}')
        return y_adj0

    def adjoint_method(self, t_eval, y_target, **kwargs):
        """
        Method to get gradients using adjoint method
        :param t_eval:
        :param y_target:
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
        aug_dyn_fun = self.construct_aug_dyn(sol, **kwargs)
        a = self.backward_solve(t_eval, dL_dy, aug_dyn_fun)
        return loss, dL_dy, a

    def usual_method(self, t_eval, y_target, **kwargs):
        """
        Method to get gradients using direct automatic differentiation
        :param t_eval:
        :param y_target:
        :param kwargs:
        :return:
        """
        with tf.GradientTape() as tape:
            sol = self.forward_solve(t_eval, y_target[0, :], **kwargs)
            y_pred = sol['y'][::self.n_ref, :]
            loss = self.loss_func(y_target, y_pred)
        dL_dp = tape.gradient(loss, self.model.trainable_variables)
        return loss, dL_dp

    def fit(self, t_eval, y_target, n_epoch=20, n_fold=5, adjoint_method=False,
            opt=tf.keras.optimizers.Adam(learning_rate=0.05), **kwargs):
        """
        Method to fit model to target data
        :param t_eval:
        :param y_target:
        :param n_epoch:
        :param n_fold:
        :param adjoint_method:
        :param opt:
        :param kwargs:
        :return:
        """
        if self.n_external > 0:
            if 'x_external' in kwargs.keys():
                x_external = kwargs['x_external']
                # create interpolation
                x_external_interp_np = interp1d(t_eval.numpy(),
                                                x_external.numpy(), kind='linear', axis=0)
                x_external_interp = lambda t: tf.expand_dims(tf.constant(x_external_interp_np(t)), axis=0)
                dict_external = {'x_external': x_external_interp}
            else:
                print('No external data')
                return None
        else:
            dict_external = {}
        #
        shuffle_folds = True
        # create folds
        if n_fold > 2:
            kf = KFold(n_splits=n_fold)
            ix_list = [ix_train for __, ix_train in kf.split(t_eval)]
        else:
            ix_list = [np.arange(0, len(t_eval))]
        loss_list = []
        # start epochs
        t_tot = time.time()
        for i in range(n_epoch):
            print(f'--- Epoch #{i + 1} ---')
            t_epoch = time.time()
            epoch_loss = 0.0
            if shuffle_folds:
                shuffle(ix_list)
            for ix_train in ix_list:
                if adjoint_method:
                    loss, dL_dy, a = self.adjoint_method(tf.gather(t_eval, indices=ix_train),
                                                         tf.gather(y_target, indices=ix_train),
                                                         **dict_external)
                    grads_p = a[0, self.n_dynamic:]
                    grads_list = self.unflatten_param(grads_p)
                else:
                    loss, grads_list = self.usual_method(tf.gather(t_eval, indices=ix_train),
                                                         tf.gather(y_target, indices=ix_train),
                                                         **dict_external)
                epoch_loss += loss
                opt.apply_gradients(zip(grads_list, self.model.trainable_variables))
                # print('Batch finished')
                # print(f'Loss: {loss}')
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
        sol2 = self.forward_solve(t_eval, y_target[0, :], **dict_external)
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(t_eval.numpy(), y_target[:, 0].numpy())
        ax.plot(sol2['t'].numpy(), sol2['y'][:, 0].numpy())
