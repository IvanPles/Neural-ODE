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

model = keras.Sequential(
    [
        keras.Input(shape=(2,)),
        layers.Dense(2, name="layer1"),
    ]
)

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
y_target = tf.constant(np.transpose(sol.y) )


n_epoch = 2
n_ode.fit(t_target, y_target, n_epoch=n_epoch, n_fold=10, adjoint_method=False)