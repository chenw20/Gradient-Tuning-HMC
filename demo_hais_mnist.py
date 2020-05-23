import tensorflow as tf
import numpy as np
import time
from core.ais import hais_gauss

import json

from core.hamInfnet_hei_nn import HamInfNetNN
from data import load_iwae_binarised_mnist_dataset
from decoder.vae_dcnn_mnist import VAE_DCNN_GPU, VAEQ_CONV, VAE_DCNN


def load_setting(model_path):
    setting_filename = model_path + 'setting.json'
    if setting_filename:
        with open(setting_filename, 'r') as f:
            setting = json.load(f)
            print('Restored setting from {}'.format(setting_filename))
    print(setting)
    return setting


def load_model(model_path, mb_size=36, dtype=tf.float32):
    setting = load_setting(model_path)
    z_dim = setting['z_dim'] 
    h_dim = setting['h_dim']
    num_layers = setting['num_layers']
    num_lfsteps = setting['num_lfsteps']
    generator = setting['generator']
    print(generator)
    if 'vfun' in setting.keys():
        vfun = setting['vfun']
    else:
        vfun = 'sigmoid'
    vae_decoder = VAE_DCNN(h_dim=h_dim, z_dim=z_dim)
    vae_decoder_pot = VAE_DCNN_GPU(h_dim=h_dim, z_dim=z_dim, gen=vae_decoder.get_generator(), afun=vfun)
    vae_encoder = VAEQ_CONV(alpha=0.3,z_dim=z_dim, h_dim=h_dim)
    hamInfNet_hm = HamInfNetNN(num_layers=num_layers,
                               num_lfsteps=num_lfsteps,
                               sample_dim=z_dim,
                               dtype=dtype)
    return vae_encoder, vae_decoder, vae_decoder_pot, hamInfNet_hm

def demo(dataset, device="GPU", dtype=tf.float32):
    model_path = "model/mb_128_layers_10_epoch_50/"
    mb_demo_size = 1

    setting = load_setting(model_path)
    X_mnist_dim = setting['X_mnist_dim']
    z_dim = setting['z_dim']

    device_config = '/device:{}:0'.format(device)
    tf.reset_default_graph()

    with tf.device(device_config):
        X_batch_demo = tf.placeholder(dtype, shape=[mb_demo_size, X_mnist_dim])
        vae_encoder, vae_decoder, vae_decoder_pot, hamInfNet_hm = load_model(model_path, mb_size=mb_demo_size)
        q0_mean, q0_logvar = vae_encoder.Q(X_batch_demo)

        pot_fun = lambda z: vae_decoder_pot.pot_fun(data_x=X_batch_demo, sample_z=z)
       
        log_partition, log_weights, sample, acp_rate = hais_gauss(pot_target=pot_fun, num_chains=1000, dim=z_dim,
                                                              num_scheduled_dists=1000,
                                                              num_leaps=5,
                                                              step_size=0.15)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model restored.")
        log_partition_list=[]
        log_weights_list=[]
        sample_list=[]
        acp_rate_list=[]
        for i in range(10000):
            start = time.time()
            X_mb_demo, _ = dataset.test.next_batch(mb_demo_size)
            log_partition_i, log_weights_i, sample_i, acp_rate_i = sess.run([log_partition, log_weights, sample, acp_rate], feed_dict={X_batch_demo: X_mb_demo})
            end = time.time()
            time_cost = end - start
            log_partition_list.append(log_partition_i)
            log_weights_list.append(log_weights_i)
            sample_list.append(sample_i)
            acp_rate_list.append(acp_rate_i)
            print('iter: {}, log_partition: {}, time: {}'.format(i+1, log_partition_i, time_cost))
    return log_partition_list, log_weights_list, sample_list, acp_rate_list

        


if __name__ == '__main__':
    mnist = load_iwae_binarised_mnist_dataset()
    log_partition_list, log_weights_list, sample_list, acp_rate_list = demo(dataset=mnist,device='CPU')
    print('test set log(x): {}'.format(np.mean(np.asarray(log_partition_list))))
    np.save('hais_result/opt_inf/log_x.npy', np.asarray(log_partition_list))
    np.save('hais_result/opt_inf/log_weights.npy', np.asarray(log_weights_list))
    np.save('hais_result/opt_inf/samples.npy', np.asarray(sample_list))
    np.save('hais_result/opt_inf/acp_rate.npy', np.asarray(acp_rate_list))
