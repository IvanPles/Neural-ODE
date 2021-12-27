import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
sys.path.append('..')
import neural_ode.NeuralODE
import neural_ode.ODESolvers

tf.keras.backend.set_floatx('float64')


class SimpleModel(tf.keras.Model):
    def __init__(self, dyn_dim = 1):
        super().__init__()
        w_init = tf.random_normal_initializer(mean=-1.0, stddev=0.05)
        self.w = tf.Variable(
            initial_value = w_init(shape=(dyn_dim*2, dyn_dim), dtype="float64"),
            trainable=True,
        )
        self.dyn_dim = dyn_dim

    def call(self, inputs):
        vels = inputs[:, self.dyn_dim:]
        accs = tf.matmul(inputs, self.w)
        return tf.concat([vels, accs], axis=1)


model = SimpleModel()

n_ode = neural_ode.NeuralODE.NeuralODE(model, 2)
#
N_n = int(2)
c = 0.1
k = 4.0

def oscilator(t, y):
    return np.array([y[1], -c*y[1]-k*y[0]])

t_final = 20.0
n_eval = int(501)
t_span = np.array([0.0, t_final])
y0 = np.array([1.0, 0.0])
sol = solve_ivp(oscilator, t_span, y0, t_eval=np.linspace(0, t_final, num=n_eval))

# transform to tensorflow
t_span_tf = tf.constant(t_span)
y0_tf = tf.constant(y0, dtype=tf.float64)
t_target = tf.constant(sol.t)
# only displacements
y_target = tf.expand_dims(tf.constant(np.transpose(sol.y[0, :])), axis=1)

#
model.variables[0].assign(np.array([[-k+0.0], [-c]]))

n_epoch = 2
n_ode.fit(t_target, y_target, n_epoch=n_epoch, n_fold=1, adjoint_method=False, missing_derivative=[0])