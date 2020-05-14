import math

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

from core.hamInfnet_hei_nn import HamInfNetNN
from data import load_iwae_binarised_mnist_dataset
from decoder.vae_dcnn_mnist import VAE_DCNN_GPU, VAEQ_CONV, VAE_DCNN
from util.utils import plot_mnist


def dybinarize_mnist(x, high=1.0):
    return (x > np.random.uniform(high=high, size=np.shape(x))).astype(np.float32)


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
    z_dim = setting['z_dim']  # 50
    h_dim = setting['h_dim']

    num_layers = setting['num_layers']
    num_lfsteps = setting['num_lfsteps']
    generator = setting['generator']
    print(generator)
    if 'vfun' in setting.keys():
        vfun = setting['vfun']
    else:
        vfun = 'sigmoid'

    #vae_decoder = VAE_DCNN(input_batch_size=mb_size, dimH=h_dim, dimZ=z_dim)
    vae_decoder = VAE_DCNN(h_dim=h_dim, z_dim=z_dim)
    vae_decoder_pot = VAE_DCNN_GPU(h_dim=h_dim, z_dim=z_dim, gen=vae_decoder.get_generator(), afun=vfun)
    #vae_encoder = VAEQ_CONV(input_batch_size=mb_size, z_dim=z_dim, h_dim=h_dim)
    vae_encoder = VAEQ_CONV(alpha=0.3,z_dim=z_dim, h_dim=h_dim)
    hamInfNet_hm = HamInfNetNN(num_layers=num_layers,
                               num_lfsteps=num_lfsteps,
                               sample_dim=z_dim,
                               dtype=dtype)

    # hamInfNet_hm = HamInfNet(num_layers=num_layers,
    #                          num_lfsteps=num_lfsteps,
    #                          sample_dim=z_dim,
    #                          training=False,
    #                          min_step_size=0.0,  # It must be 0.0 otherwise the step size is not the same as
    #                          # the imported models
    #                          dtype=dtype)
    return vae_encoder, vae_decoder, vae_decoder_pot, hamInfNet_hm


def demo(dataset, device="CPU", dtype=tf.float32):
    model_path = "model/debug_hei_vae/"

    mb_demo_size = 36
    demo_size = (6, 6)
    setting = load_setting(model_path)
    X_mnist_dim = setting['X_mnist_dim']
    z_dim = setting['z_dim']

    device_config = '/device:{}:0'.format(device)
    tf.reset_default_graph()

    with tf.device(device_config):
        X_batch_demo = tf.placeholder(dtype, shape=[mb_demo_size, X_mnist_dim])
        vae_encoder, vae_decoder, vae_decoder_pot, hamInfNet_hm = load_model(model_path, mb_size=mb_demo_size)
        print(vae_encoder.get_parameters())
        print(vae_decoder.get_parameters())
        q0_mean, q0_logvar = vae_encoder.Q(X_batch_demo)

        z_prior = tf.random_normal(shape=(mb_demo_size, z_dim), dtype=dtype)

        PX_demo_gen = vae_decoder.nlog_px_z(z_prior)[0]
        pot_fun = lambda z: vae_decoder_pot.pot_fun(data_x=X_batch_demo, sample_z=z)
        z_recon_samples = hamInfNet_hm.build_simulation_graph(pot_fun=pot_fun, mean=q0_mean, logvar=q0_logvar,
                                                              sample_batch_size=1,
                                                              input_data_batch_size=mb_demo_size)
        PX_demo_recon = vae_decoder.nlog_px_z(tf.reduce_mean(z_recon_samples, axis=0))[0]
        # X_demo_gen_samples_valid, _ = vae.nlog_px_z(z_post_valid)
        # Z_demo_samples = hamInfNet_hm.build_simulation_gauss_graph(pot_fun_demo, demo_sample_batch_size,
        #                                                            mb_test_demo_size)
        #
        # X_demo_recover_samples, _ = vae.nlog_px_z(Z_demo_samples)
        # pot_recon, recon_loss, prior_loss = pot_fun_demo_debug(Z_demo_samples)
        # X_demo_sample_recon_dist = tf.reduce_mean(X_demo_recover_samples, axis=0)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model restored.")

        for _ in range(5):
            X_mb_demo, _ = dataset.test.next_batch(mb_demo_size)
            # q0_mean_value, q0_logvar_value, lfstep_size_value = sess.run([q0_mean, q0_logvar, hamInfNet_hm.lfstep_size], feed_dict={X_batch_demo: X_mb_demo})
            # print(np.shape(q0_mean_value))
            # print(np.shape(q0_logvar_value))
            # # print(lfstep_size_value)

            X_gen_samples, PX_demo_recon_samples = sess.run([PX_demo_gen, PX_demo_recon], feed_dict={X_batch_demo: X_mb_demo})
            plot_mnist(X_gen_samples, size=demo_size, title="gen", save_path=None)
            plot_mnist(X_mb_demo, size=demo_size, title="test_origin", save_path=None)
            plot_mnist(PX_demo_recon_samples, size=demo_size, title="recon", save_path=None)

            plt.show()


if __name__ == '__main__':
    mnist = load_iwae_binarised_mnist_dataset()
    demo(dataset=mnist)


