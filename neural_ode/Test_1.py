import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
sys.path.append('..')
import neural_ode.NeuralODE

import neural_ode.ODESolvers

tf.keras.backend.set_floatx('float64')

model = keras.Sequential(
    [
        keras.Input(shape=(2,)),
        layers.Dense(2, name="layer1"),
    ]
)

n_ode = neural_ode.NeuralODE.NeuralODE(model, 2,
                                       solver=neural_ode.ODESolvers.TaylorSecondOrder())
c = 0.1
k = 4.0

n_ode.model.variables[0].assign(np.array([[0.0, -k], [1.0, -c]]))
n_ode.model.variables[1].assign(np.array([0,0]))

y0 = tf.constant([[3.0, 4.0]], dtype=tf.float64)
resid = lambda y: n_ode.model(y) - tf.constant([[2.0, 3.0]], dtype=tf.float64)


y0_tf = tf.constant([1.0, 0.0], dtype=tf.float64)
sol, _ = n_ode.solver(n_ode.ode_wrap, tf.constant([0.0, 0.1], dtype=tf.float64), tf.expand_dims(y0_tf, axis=0),
                            step_size=0.05, jac = n_ode.grad_inps_wrap)

print(sol)
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
y_target = tf.constant(np.transpose(sol.y) )


n_epoch = 40
n_ode.fit(t_target, y_target, n_epoch=n_epoch, n_fold=10, adjoint_method=False, conjugate=True, )