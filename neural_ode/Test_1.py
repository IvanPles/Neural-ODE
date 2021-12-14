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
                                       solver=neural_ode.ODESolvers.BackwardEulerMethod())
c = 0.1
k = 4.0

n_ode.model.variables[0].assign(np.array([[0.0, -k], [1.0, -c]]))
n_ode.model.variables[1].assign(np.array([0,0]))

y0 = tf.constant([[3.0, 4.0]], dtype=tf.float64)
resid = lambda y: n_ode.model(y) - tf.constant([[2.0, 3.0]], dtype=tf.float64)

#def jac1(x):
#    with tf.GradientTape() as tape:
#        tape.watch(x)
#        y = n_ode.model(x)
#    J = tape.batch_jacobian(y, x)
#    return J[0,:,:]
#sol11 = neural_ode.ODESolvers.newtons_method(y0, resid, jac1)

y0_tf = tf.constant([1.0, 0.0], dtype=tf.float64)
jac1 = lambda y: n_ode.grad_inps(y)[0,:,:]
sol, _ = n_ode.solver(n_ode.ode_wrap, tf.constant([0.0, 0.1], dtype=tf.float64), tf.expand_dims(y0_tf, axis=0),
                            step_size=0.05, jac = jac1)

print(sol)