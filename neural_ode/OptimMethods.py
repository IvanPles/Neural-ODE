import tensorflow as tf


class ConjugateGradientMethod:

        def __init__(self, **kwargs):
            default_options = {'n_restart': 10}
            if 'n_restart' in kwargs.keys():
                self.n_restart = kwargs['n_restart']
            else:
                self.n_restart = default_options['n_restart']
            self.prev_d = None
            self.prev_grad = None
            self.learning_rate = 0.1
            self.iter = 0

        def apply_gradients(self, grads):
            flatten_grad = [tf.reshape( x, (tf.size(x)) ) for x in grads]
            flatten_grad = tf.concat(flatten_grad, axis=0)
            if self.prev_d is None:
                self.prev_d = tf.zeros(tf.size(flatten_grad))
                self.prev_grad = tf.zeros(tf.size(flatten_grad))
            om = tf.matmul((flatten_grad - self.prev_grad)*tf.transpose(flatten_grad))/tf.norm(flatten_grad)
            new_dir = -flatten_grad+om*self.prev_d
