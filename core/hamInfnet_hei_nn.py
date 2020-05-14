import tensorflow as tf

from core.ais import hais_gauss
from util.constant import log_2pi
from core.ham import leapfrog
import numpy as np


class HamInfNetNN:
    def __init__(self, num_lfsteps,
                 num_layers,
                 sample_dim,
                 min_step_size=0.01,
                 dtype=tf.float32):
        self.num_lfsteps = num_lfsteps
        self.num_layers = num_layers
        self.num_layers_max = num_layers

        self.sample_dim = sample_dim
        self.dtype = dtype

        self.lfstep_size_raw = tf.get_variable(name="lfstep_size",
                                           initializer=tf.constant(
                                               np.random.uniform(0.02, 0.05, size=(num_layers, 1, sample_dim)),
                                               dtype=dtype),
                                           trainable=True, dtype=dtype)
        self.lfstep_size = tf.abs(self.lfstep_size_raw) + min_step_size
        self.log_r_var = tf.get_variable(name="log_r_var",
                                         shape=(num_layers, 1, sample_dim),
                                         initializer=tf.zeros_initializer,
                                         trainable=True, dtype=dtype)
        self.momentum = tf.exp(0.5* self.log_r_var)* \
            tf.random_normal(shape=self.log_r_var.shape)
        """
        self.momentum = tf.random_normal(stddev=tf.sqrt(tf.exp(self.log_r_var)),
                                    shape=self.log_r_var.shape)
        """
        
        self.log_inflation = tf.get_variable(name="log_inflation",
                                             shape=(),
                                             initializer=tf.zeros_initializer,
                                             trainable=True, dtype=dtype)

    def getParams(self):
        return self.lfstep_size_raw, self.log_r_var, self.log_inflation, self.q0_mean, self.log_q0_std

    def __build_LF_graph(self, pot_fun, state_init, back_prop=False):
        cond = lambda layer_index, state: tf.less(layer_index, self.num_layers)

        def _loopbody(layer_index, state):
            state_new, _ = leapfrog(x=state,
                                    r=self.momentum[layer_index],
                                    pot_fun=pot_fun,
                                    eps=self.lfstep_size[layer_index],
                                    r_var=tf.exp(self.log_r_var[layer_index]),
                                    numleap=self.num_lfsteps,
                                    back_prop=back_prop)
            return layer_index + 1, state_new

        _, state_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init))
        return state_final

    def build_simulation_graph(self, pot_fun, mean, logvar, sample_batch_size, input_data_batch_size):
        state_init = self.state_init_gen(mean, logvar, sample_batch_size, input_data_batch_size)
        state_samples = self.__build_LF_graph(pot_fun, state_init, back_prop=False)
        return state_samples

    def state_init_gen(self, mean, logvar, sample_batch_size, input_data_batch_size):
        q0_init_shape = (sample_batch_size, input_data_batch_size, self.sample_dim)
        q0_init = tf.random_normal(shape=q0_init_shape, dtype=self.dtype)
        state_init = q0_init * tf.exp(self.log_inflation)*(tf.exp(logvar/2)) + mean
        return state_init

    def build_elbo_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size, training=False):
        """
        sample batch shape: sample_batch_size x input_data_batch_size x sample_dim
        potential batch shape: sample_batch_size x input_data_batch_size

        the list of shape of variables
        state_init: sample batch shape
        pot_energy_all_samples_final: potential batch shape
        log_q0_z: potential batch shape

        :param pot_fun: the potential function takes a batch of samples as input and outputs batch of potential values
        :param state_init_gen: take sample_batch_size and input_data_batch_size as input and outputs a batch of samples
        from initial distribution q_0 and their log probability log q_0
        :param sample_batch_size: the number of samples used to estimate the gradient
        :param input_data_batch_size: the batch size of input data, which must be compatible with potential function
        :param training: Boolean variable true for training / false for evaluation

        :return:
        elbo_mean: the Monte Carlo (sample average) estimation of loss function

        """
        # state_init shape: sample_batch_size x input_data_batch_size x sample dimensions
        # log_q_z shape: sample_batch_size x input_data_batch_size
        state_init = state_init_gen(sample_batch_size, input_data_batch_size)
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers_max, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        state_init_stop_gradient = tf.stop_gradient(state_init)
        state_final = self.__build_LF_graph(pot_fun, state_init_stop_gradient, momentum, back_prop=training)

        ####################################### Compute Energy ########################################
        # pot_energy_all_sample shape: sample_batch_size x input_data_batch_size
        pot_energy_all_samples_final = pot_fun(state_final)  # potential function is the negative log likelihood
        return tf.reduce_mean(pot_energy_all_samples_final)