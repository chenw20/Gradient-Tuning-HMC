import tensorflow as tf

from util.constant import log_2pi
from core.ham import leapfrog
import numpy as np


class HamInfNetHEI:
    def __init__(self, num_lfsteps,
                 num_layers,
                 sample_dim,
                 training=False,
                 min_step_size=0.01,
                 #min_step_size = 0.0,
                 init_step_scale=1.0,
                 log_q0_std_init=0.0,
                 name_space="",
                 stop_gradient=True,
                 dtype=tf.float32):
        self.num_lfsteps = num_lfsteps
        self.num_layers = num_layers
        self.num_layers_max = num_layers
        self.stop_gradient = stop_gradient
        self.sample_dim = sample_dim
        self.dtype = dtype

        self.lfstep_size_raw = tf.get_variable(name="{}lfstep_size".format(name_space),
                                               initializer=tf.constant(
                                                   init_step_scale*np.random.uniform(0.02,0.05,size=(num_layers,1,sample_dim)),
                                                   dtype=dtype),
                                               trainable=True, dtype=dtype)
        """
        self.q0_mean = tf.get_variable(name="{}q0_mean".format(name_space),
                                       shape=(1, sample_dim),
                                       initializer=tf.zeros_initializer,
                                       trainable=training, dtype=dtype)
        self.log_q0_std = tf.get_variable(name="{}log_q0_std".format(name_space),
                                          shape=(1, sample_dim),
                                          initializer=tf.constant_initializer(value=log_q0_std_init),
                                          trainable=training, dtype=dtype)
        """
        self.lfstep_size = tf.abs(self.lfstep_size_raw) + min_step_size
        self.log_r_var = tf.get_variable(name="{}log_r_var".format(name_space),
                                         shape=(num_layers, 1, sample_dim),
                                         initializer=tf.zeros_initializer,
                                         trainable=False, dtype=dtype)
        self.momentum = tf.exp(0.5* self.log_r_var)* \
            tf.random_normal(shape=self.log_r_var.shape)
        
        self.raw_inflation = tf.get_variable(name="{}log_inflation".format(name_space),
                                             shape=(),
                                             initializer=tf.zeros_initializer,
                                             trainable=True, dtype=dtype)
        self.inflation=tf.abs(self.raw_inflation)
        
    def __build_LF_graph_hmc(self, pot_fun, state_init, num_layers=None, back_prop=False):
        if num_layers is None:
            num_layers_effect = self.num_layers
        else:
            num_layers_effect = tf.minimum(num_layers, self.num_layers)
        cond = lambda layer_index, state: tf.less(layer_index, num_layers_effect)

        def _loopbody(layer_index, state):
            state_new, _ = leapfrog(x=state,
                                    r=self.momentum[layer_index],
                                    pot_fun=pot_fun,
                                    eps=self.lfstep_size[layer_index],
                                    r_var=tf.exp(self.log_r_var[layer_index]),
                                    numleap=self.num_lfsteps,
                                    stop_gradient_pot=self.stop_gradient,
                                    back_prop=back_prop)
            return layer_index + 1, state_new

        _, state_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init))
        return state_final
    
    def __build_LF_graph_ksd(self, pot_fun, state_init, num_layers=None, back_prop=False):
        if num_layers is None:
            num_layers_effect = self.num_layers
        else:
            num_layers_effect = tf.minimum(num_layers, self.num_layers)
        cond = lambda layer_index, state: tf.less(layer_index, num_layers_effect)

        def _loopbody(layer_index, state):
            state_new, _ = leapfrog(x=state,
                                    r=tf.stop_gradient(self.momentum[layer_index]),
                                    pot_fun=pot_fun,
                                    eps=tf.stop_gradient(self.lfstep_size[layer_index]),
                                    r_var=tf.exp(tf.stop_gradient(self.log_r_var[layer_index])),
                                    numleap=self.num_lfsteps,
                                    stop_gradient_pot=self.stop_gradient,
                                    back_prop=back_prop)
            return layer_index + 1, state_new

        _, state_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(0, state_init))
        return state_final

    
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
        state_init, log_q0_z = state_init_gen(sample_batch_size, input_data_batch_size, self.inflation)
        """
        momentum = tf.random_normal(stddev=tf.sqrt(tf.exp(self.log_r_var)),
                                    shape=(self.num_layers_max, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        """
        # state_init_stop_gradient = tf.stop_gradient(state_init)
        #state_final = self.__build_LF_graph(pot_fun, state_init, momentum, back_prop=training)
        state_final = self.__build_LF_graph_hmc(pot_fun, state_init, back_prop=training)
        # momentum_init = momentum[0]
        # momentum_last = tf.random_normal(stddev=1.0,
        #                                  shape=(sample_batch_size, input_data_batch_size,
        #                                         self.sample_dim), dtype=self.dtype)
        # state_final2, momentum_final = leapfrog(x=state_final,
        #          r=momentum_last,
        #          pot_fun=pot_fun,
        #          eps=self.lfstep_size[self.num_layers-1],
        #          r_var=1.0,
        #          numleap=self.num_lfsteps*5,
        #          back_prop=training)
        # state_final = state_final2
        #
        # elbo_per_data_momentum = -0.5 * tf.reduce_sum(momentum_final ** 2 - momentum_init ** 2, axis=-1)

        ####################################### Compute Energy ########################################
        # pot_energy_all_sample shape: sample_batch_size x input_data_batch_size
        pot_energy_all_samples_final = pot_fun(state_final)  # potential function is the negative log likelihood
        #pot_energy_all_samples_init = pot_fun(state_init)  # potential function is the negative log likelihood
        pot_gaussian_prior = 0.5*tf.reduce_sum(state_final**2 + log_2pi, axis=-1)

        # nelbo_per_sample = -elbo_per_data_momentum
        nelbo_per_sample = pot_energy_all_samples_final
        # nelbo_per_sample = pot_energy_all_samples_final + pot_energy_all_samples_init + log_q0_z  #- elbo_per_data_momentum #+ kinetic_out - kinetic_in  #  # -logp(x) + logq(x)
        #nelbo_per_sample_x = pot_energy_all_samples_final
        elbo_per_data = tf.reduce_mean(-nelbo_per_sample, axis=0)
        #elbo_per_data_x = tf.reduce_mean(-nelbo_per_sample_x, axis=0)
        #logD_per_data = tf.reduce_logsumexp(-nelbo_per_sample, axis=0, keepdims=True) - tf.log(
        #    tf.constant(sample_batch_size, dtype=tf.float32))
        elbo_mean = tf.reduce_mean(elbo_per_data)
        #elbo_x_mean = (-tf.reduce_mean(elbo_per_data_x), tf.reduce_mean(log_q0_z))
        #logD_mean = tf.reduce_mean(logD_per_data)
        recon_mean = tf.reduce_mean(pot_energy_all_samples_final - pot_gaussian_prior)
        #return elbo_mean, recon_mean, elbo_x_mean   # elbo_mean is actually neg_elbo_mean
        return elbo_mean, recon_mean      

    def build_ksd_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size, training=False):
        # state_init shape: sample_batch_size x input_data_batch_size x sample dimensions
        # log_q_z shape: sample_batch_size x input_data_batch_size
        state_init, log_q0_z = state_init_gen(sample_batch_size, input_data_batch_size, self.inflation)
        
        state_final = self.__build_LF_graph_ksd(pot_fun, state_init, back_prop=training)
        
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
            h_square = tf.stop_gradient(get_median(pdist_square))
            #h_square = tf.stop_gradient(tf.reduce_mean(pdist_square))
            Kxy = tf.exp(- pdist_square / (2* h_square))
        
            # now compute KSD
            Sqxdy = tf.matmul(tf.stop_gradient(Sqx), tf.transpose(z)) -\
                tf.tile(tf.reduce_sum(tf.stop_gradient(Sqx) * z, 1, keepdims=True), (1, K))
            Sqxdy = -Sqxdy / (h_square)
        
            dxSqy = tf.transpose(Sqxdy)
            dxdy = -pdist_square / (h_square ** 2) + dimZ.value / (h_square)
            # M is a (K, K) tensor
            M = (tf.matmul(tf.stop_gradient(Sqx), tf.transpose(tf.stop_gradient(Sqx))) +\
                 Sqxdy + dxSqy + dxdy) * Kxy
            
            # the following for U-statistic
            if flag_U:
                M2 = M - tf.diag(tf.diag(M))
                return tf.reduce_sum(M2) / (K.value * (K.value-1) )
            
            # the following for V-statistic
            return tf.reduce_mean(M) 
        """
        def KSD_no_second_gradient(z, Sqx):
            def imq_kernel(x,dim, beta=-.5, c=1.):
                #dim = tf.cast(tf.shape(x)[-1],tf.float32)
                XY = tf.matmul(x, tf.transpose(x))
                X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
                X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
                pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix
                
                def get_median(v):
                    v = tf.reshape(v, [-1])
                    if v.get_shape()[0] % 2 == 1:
                        mid = v.get_shape()[0]//2 + 1
                        return tf.nn.top_k(v, mid).values[-1]
                    else:
                        mid1 = v.get_shape()[0]//2
                        mid2 = v.get_shape()[0]//2 + 1
                        return 0.5* (tf.nn.top_k(v, mid1).values[-1]+tf.nn.top_k(v, mid2).values[-1])
                h_square=tf.stop_gradient(tf.clip_by_value(get_median(pdist), clip_value_min=1e-4,clip_value_max=1e16))
                kxy = (c + pdist/(h_square)) ** beta

                coeff = 2 * beta * ((c + pdist/(h_square)) ** (beta-1))/(h_square)
                dxkxy = tf.matmul(coeff, x) - tf.multiply(x, tf.expand_dims(tf.reduce_sum(coeff, axis=1), 1))

                dxykxy_tr = tf.multiply((c + pdist/(h_square)) ** (beta - 2),
                                        - 2 * dim * c * beta/(h_square) + (- 4 * beta ** 2 + (4 - 2 * dim) * beta)/(tf.clip_by_value(h_square**2,clip_value_min=1e-4,clip_value_max=1e16)) * pdist)

                return kxy, dxkxy, dxykxy_tr
            def ksd_emp(x,Sqx):
                dim = tf.cast(tf.shape(x)[-1],tf.float32)
                #sq = tf.stop_gradient(Sqx)
                sq = Sqx
                kxy, dxkxy, dxykxy_tr = imq_kernel(x,dim)
                t13 = tf.multiply(tf.matmul(sq, tf.transpose(sq)), kxy) + dxykxy_tr
                t2 = 2 * tf.trace(tf.matmul(sq, tf.transpose(dxkxy)))
                n = tf.cast(tf.shape(x)[0], tf.float32)

                # ksd = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))
                ksd = (tf.reduce_sum(t13) + t2) / (n ** 2)

                return ksd
            return ksd_emp(z,tf.stop_gradient(Sqx))
        
        """  
        
        # Now apply KSD function to each input in the batch
        # pot_fun is neg-log-lik
        pot_energy_all_samples = pot_fun(state_final)  # sample_size * input_batch , neg log-lik
        grad_pot_all_samples = tf.gradients(ys= -pot_energy_all_samples, xs = state_final)[0] #sample_size * input_batch* latent_dim
        """
        cond = lambda batch_index, ksd_sum: tf.less(batch_index, input_data_batch_size)
        def _loopbody(batch_index, ksd_sum):
            return batch_index + 1, ksd_sum + KSD_no_second_gradient(state_final[:,batch_index,:], grad_pot_all_samples[:,batch_index,:])
        
        
        _, ksd_sum_final = tf.while_loop(cond=cond, body=_loopbody, loop_vars=(1, KSD_no_second_gradient(state_final[:,0,:],grad_pot_all_samples[:,0,:])))
        return ksd_sum_final/input_data_batch_size
        """
        return KSD_no_second_gradient(tf.reshape(state_final,[sample_batch_size,-1]),  tf.reshape(grad_pot_all_samples,[sample_batch_size,-1]))
        
    """
    def build_ksd_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size, training=False):
        # state_init shape: sample_batch_size x input_data_batch_size x sample dimensions
        # log_q_z shape: sample_batch_size x input_data_batch_size
        state_init, log_q0_z = state_init_gen(sample_batch_size, input_data_batch_size, self.log_inflation)
        
        state_final = self.__build_LF_graph_ksd(pot_fun, state_init, back_prop=training)
        
        def KSD_no_second_gradient(z, Sqx):
            # dim_z is input_batch * sample_size * latent_dim 
            # compute the rbf kernel
            input_batch_size, K, dimZ = z.shape
            pdist_square = tf.reduce_sum((z[:,None,:]-z[:,:,None])**2, -1)
            
            
            
            h_square = tf.stop_gradient(tf.reduce_mean(pdist_square,(-1,-2)))
            
            #Kxy = tf.exp(- pdist_square / (2* h_square) )
            Kxy = tf.exp(tf.einsum('ijk,i->ijk', -pdist_square, 1./h_square)/ 2.0)
            #Kxy = tf.exp(- tf.exp(tf.log(pdist_square+0.0000001)-tf.log(2* median+0.0000001)))
        
            # now compute KSD

            Sqxdy= tf.einsum('ijk,ikl->ijl',Sqx,tf.einsum('ijk->ikj',z))-tf.tile(tf.reduce_sum(Sqx*z,-1,keepdims=True),(1,1,K))
            Sqxdy = -Sqxdy/tf.expand_dims(tf.expand_dims(h_square,-1),-1)
            #dxSqy = tf.transpose(Sqxdy)
            dxSqy = tf.einsum('ijk->ikj',Sqxdy)
            #dxdy = -pdist_square / (h_square ** 2) + dimZ.value / h_square
            dxdy = tf.einsum('ijk,i->ijk', -pdist_square, 1./(h_square)**2) + tf.expand_dims(tf.expand_dims(dimZ.value / h_square, -1),-1)
            #dxdy = -pdist_square / (h_square ** 2) 
            # M is a (input_batch, K, K) tensor

            M = (tf.einsum('ijk,ikl->ijl',Sqx,tf.einsum('ijk->ikj',Sqx))+Sqxdy + dxSqy + dxdy) * Kxy
            # the following for V-statistic
            #M2 = M - np.diag(np.diag(M))
            #return tf.reduce_sum(M) / (K.value * K.value )
            return tf.reduce_mean(M) 
        
        # Now apply KSD function to each input in the batch
        # pot_fun is neg-log-lik
        pot_energy_all_samples = -pot_fun(state_final)  # log_lik,   sample_size * input_batch , neg log-lik
        grad_pot_all_samples = tf.gradients(ys= pot_energy_all_samples, xs = state_final)[0] #sample_size * input_batch* latent_dim
        return KSD_no_second_gradient(tf.einsum('ijk->jik',state_final), tf.einsum('ijk->jik',grad_pot_all_samples))
    """
    def getParams(self):
        return self.lfstep_size_raw, self.log_r_var, self.raw_inflation
    
    def getInflation(self):
        return self.inflation
    """
    def getlf_step(self):
        return self.lfstep_size_raw

    def getInitParams(self):
        return self.q0_mean, self.log_q0_std
    
    def __build_LF_scan_graph(self, pot_fun, state_init, momentum, back_prop=False):
        initializer = (state_init, tf.zeros_like(state_init, dtype=self.dtype))
        elems = tf.range(0, self.num_layers_max)

        def _loopbody(state, layer_index):
            state_new, momentum_new = leapfrog(x=state[0],
                                               r=momentum[layer_index],
                                               pot_fun=pot_fun,
                                               eps=self.lfstep_size[layer_index],
                                               r_var=1.0,
                                               numleap=self.num_lfsteps,
                                               stop_gradient_pot=self.stop_gradient,
                                               back_prop=back_prop)
            return (state_new, momentum_new)

        state_final = tf.scan(_loopbody, elems=elems, initializer=initializer)
        final_position = state_final[0][-1]
        momentum_out = state_final[1]
        return final_position, momentum_out
    
    def build_simulation_gauss_graph(self, pot_fun, sample_batch_size, input_data_batch_size):
        state_init, _ = self.__prepare_state_init(sample_batch_size, input_data_batch_size)
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        state_samples = self.__build_LF_graph_hmc(pot_fun, state_init, momentum, back_prop=False)
        return state_samples

    def build_simulation_gauss_graph_with_length(self, pot_fun, sample_batch_size, input_data_batch_size, length):
        state_init, _ = self.__prepare_state_init(sample_batch_size, input_data_batch_size)
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        state_samples = self.__build_LF_graph(pot_fun, state_init, momentum, length, back_prop=False)
        return state_samples

    def build_simulation_graph(self, pot_fun, state_init_gen, sample_batch_size, input_data_batch_size):
        state_init, _ = state_init_gen(sample_batch_size, input_data_batch_size)
        momentum = tf.random_normal(stddev=1.0,
                                    shape=(self.num_layers, sample_batch_size, input_data_batch_size,
                                           self.sample_dim), dtype=self.dtype)
        state_samples = self.__build_LF_graph(pot_fun, state_init, momentum, back_prop=False)
        return state_samples
    
    def __prepare_state_init(self, sample_batch_size, input_data_batch_size):
        q0_init_shape = (sample_batch_size, input_data_batch_size, self.sample_dim)
        q0_init = tf.random_normal(shape=q0_init_shape, dtype=self.dtype)
        state_init = q0_init * tf.exp(self.log_q0_std) + self.q0_mean
        log_q0_z = -0.5 * tf.reduce_sum(q0_init ** 2 + 2*self.log_q0_std, axis=-1)
        return state_init, log_q0_z  # sample_batch_size x input_data_batch_size x sample_dim

    def build_elbo_graph_gauss(self, pot_fun, sample_batch_size, input_data_batch_size, training=False):
        return self.build_elbo_graph(pot_fun, self.__prepare_state_init, sample_batch_size, input_data_batch_size,
                                     training)
    """
