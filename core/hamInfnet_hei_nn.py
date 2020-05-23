import tensorflow as tf

#from core.ais import hais_gauss
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
        
        self.log_inflation = tf.get_variable(name="log_inflation",
                                             shape=(),
                                             initializer=tf.zeros_initializer,
                                             trainable=True, dtype=dtype)

    def getParams(self):
        return self.lfstep_size_raw, self.log_r_var, self.log_inflation, self.q0_mean, self.log_q0_std
    
    def getInflation(self):
        return tf.exp(self.log_inflation)

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
    
    def build_elbo_graph(self, pot_fun, mean, logvar, sample_batch_size, input_data_batch_size, training=False):
    
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
        q0_init_shape = (sample_batch_size, input_data_batch_size, self.sample_dim)
        q0_init = tf.random_normal(shape=q0_init_shape, dtype=self.dtype)
        state_init = q0_init * tf.exp(self.log_inflation)*(tf.exp(logvar/2)) + mean
        #state_init = state_init_gen(sample_batch_size, input_data_batch_size)
        """
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers_max, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        """
        state_init_stop_gradient = tf.stop_gradient(state_init)
        state_final = self.__build_LF_graph(pot_fun, state_init_stop_gradient, back_prop=training)

        ####################################### Compute Energy ########################################
        # pot_energy_all_sample shape: sample_batch_size x input_data_batch_size
        pot_energy_all_samples_final = pot_fun(state_final)  # potential function is the negative log likelihood
        pot_gaussian_prior = 0.5*tf.reduce_sum(state_final**2 + log_2pi, axis=-1)
        recon_mean = tf.reduce_mean(pot_energy_all_samples_final - pot_gaussian_prior)
        return tf.reduce_mean(pot_energy_all_samples_final), recon_mean
    
    def build_ksd_graph(self, pot_fun, mean, logvar, sample_batch_size, input_data_batch_size, training=False):
        # state_init shape: sample_batch_size x input_data_batch_size x sample dimensions
        # log_q_z shape: sample_batch_size x input_data_batch_size
        q0_init_shape = (sample_batch_size, input_data_batch_size, self.sample_dim)
        q0_init = tf.random_normal(shape=q0_init_shape, dtype=self.dtype)
        state_init = q0_init * tf.exp(self.log_inflation)*(tf.exp(logvar/2)) + mean
        #state_init = state_init_gen(sample_batch_size, input_data_batch_size)
        """
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers_max, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        """
        state_init_stop_gradient = tf.stop_gradient(state_init)
        state_final = self.__build_LF_graph(pot_fun, state_init_stop_gradient, back_prop=training)
        
        def KSD_no_second_gradient(z, Sqx, flag_U=False):
            # dim_z is sample_size * latent_dim 
            # compute the rbf kernel
            K, dimZ = z.shape
            r = tf.reduce_sum(z*z, 1)
            # turn r into column vector
            r = tf.reshape(r, [-1, 1])
            pdist_square = r - 2*tf.matmul(z, tf.transpose(z)) + tf.transpose(r)
            
            def get_median(v):
                v = tf.reshape(v, [-1])
                if v.get_shape()[0] % 2 == 1:
                    mid = v.get_shape()[0]//2 + 1
                    return tf.nn.top_k(v, mid).values[-1]
                else:
                    mid1 = v.get_shape()[0]//2
                    mid2 = v.get_shape()[0]//2 + 1
                    return 0.5* (tf.nn.top_k(v, mid1).values[-1]+tf.nn.top_k(v, mid2).values[-1])
            h_square = get_median(pdist_square)
            Kxy = tf.exp(- pdist_square / (2* h_square) )
        
            # now compute KSD
            Sqxdy = tf.matmul(tf.stop_gradient(Sqx), tf.transpose(z)) -\
                tf.tile(tf.reduce_sum(tf.stop_gradient(Sqx) * z, 1, keepdims=True), (1, K))
            Sqxdy = -Sqxdy / h_square
        
            dxSqy = tf.transpose(Sqxdy)
            dxdy = -pdist_square / (h_square ** 2) + dimZ.value / h_square
            # M is a (K, K) tensor
            M = (tf.matmul(tf.stop_gradient(Sqx), tf.transpose(tf.stop_gradient(Sqx))) +\
                 Sqxdy + dxSqy + dxdy) * Kxy
            
            # the following for U-statistic
            if flag_U:
                M2 = M - tf.diag(tf.diag(M))
                return tf.reduce_sum(M2) / (K.value * (K.value-1) )
            
            # the following for V-statistic
            return tf.reduce_mean(M) 
        
        # Now apply KSD function to each input in the batch
        # pot_fun is neg-log-lik
        pot_energy_all_samples = pot_fun(state_final)  # sample_size * input_batch , neg log-lik
        grad_pot_all_samples = tf.gradients(ys= -pot_energy_all_samples, xs = state_final)[0] #sample_size * input_batch* latent_dim
        
        cond = lambda batch_index, ksd_sum: tf.less(batch_index, input_data_batch_size)
        def _loopbody(batch_index, ksd_sum):
            return batch_index + 1, ksd_sum + KSD_no_second_gradient(state_final[:,batch_index,:], grad_pot_all_samples[:,batch_index,:])
        
        _, ksd_sum_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(1, KSD_no_second_gradient(state_final[:,0,:],grad_pot_all_samples[:,0,:])))
        return ksd_sum_final/input_data_batch_size
