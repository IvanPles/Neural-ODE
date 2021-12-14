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

tf.keras.backend.set_floatx('float64')

model = keras.Sequential(
    [
        keras.Input(shape=(3,)),
        layers.Dense(2, name="layer1"),
    ]
)

n_ode = neural_ode.NeuralODE.NeuralODE(model, 2, n_external=1)
#
#
N_n = int(2)
c = 0.4
k = 4.0
def oscilator(t, y, f):
    f_val = f(t)
    return np.array([y[1], -c*y[1]-k*y[0]+f_val])
t_final = 200.0
n_eval = int(1000)
t_span = np.array([0.0, t_final])
t_eval = np.linspace(0, t_final, num=n_eval)
y0 = np.array([1.0, 0.0])
#
f_ext = np.vstack(( np.zeros((200,1)), np.ones((400,1))*0.5, np.ones((400,1)) ) )
f_interp = interp1d(t_eval, f_ext, kind='linear', axis=0)
#
func_1 = lambda t,y: oscilator(t,y,f_interp)
sol = solve_ivp(func_1, t_span, y0, t_eval=t_eval)
#
# transform to tensorflow
t_span_tf = tf.constant(t_span)
y0_tf = tf.constant(y0, dtype=tf.float64)
#
t_target = tf.constant(sol.t)
y_target = tf.constant(np.transpose(sol.y) )
f_ext_tf = tf.constant(f_ext)
#
# interpolation in tf
f_ext_interp_np = interp1d(t_eval, f_ext, kind='linear', axis=0)
f_ext_interp = lambda t: tf.expand_dims(tf.constant(f_ext_interp_np(t)), axis=0)
print(f_ext_interp(tf.constant([1.0])))
#
sol1 = n_ode.forward_solve(t_target, y_target[0,:], x_external=f_ext_interp)
print(sol1['y'])
#
loss, dl_dy, a = n_ode.adjoint_method(t_target[0:3], y_target[0:3,:], x_external=f_ext_interp)