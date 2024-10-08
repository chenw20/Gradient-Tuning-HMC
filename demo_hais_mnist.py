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
    vae_encoder = VAEQ_CONV(alpha=1.0,z_dim=z_dim, h_dim=h_dim)
    hamInfNet_hm = HamInfNetNN(num_layers=num_layers,
                               num_lfsteps=num_lfsteps,
                               sample_dim=z_dim,
                               dtype=dtype)
    return vae_encoder, vae_decoder, vae_decoder_pot, hamInfNet_hm

def demo(dataset, device="GPU", dtype=tf.float32):
    #model_path="model/inflation_eq1/"
    #model_path="model/inflation_eqsq2/"
    #model_path="model/inflation_eq2/"
    #model_path="model/inflation_eqsq8/"
    #model_path="../experiment/HEI/model/inflation_eq4/"
    #model_path="../experiment/HEI/model/inflation_eqsq32/"
    #model_path="../experiment/HEI/model/inflation_eqsq64/"
    
    
    #model_path="model/inflation_eqsq128/"
    #model_path="../experiment/HEI/model/inflation_eqsq256/"
    #model_path="../experiment/HEI/model/inflation_eqsq512/"
    #model_path="model/inflation_eqsq1024/"
    
    model_path="model/hei_maxsksd_alpha1/"
    #model_path="model/hei_ksd_alpha1/"
    #model_path="../experiment/HEI/model/vanila_iwae/"

    mb_demo_size = 100

    setting = load_setting(model_path)
    X_mnist_dim = setting['X_mnist_dim']
    z_dim = setting['z_dim']

    device_config = '/device:{}:0'.format(device)
    tf.reset_default_graph()

    with tf.device(device_config):
        X_batch_demo = tf.placeholder(dtype, shape=[mb_demo_size, X_mnist_dim])
        vae_encoder, vae_decoder, vae_decoder_pot, hamInfNet_hm = load_model(model_path, mb_size=mb_demo_size)
        
        q0_mean, q0_logvar = vae_encoder.Q(X_batch_demo)

        pot_fun = lambda z: vae_decoder_pot.pot_fun_train(data_x=X_batch_demo, sample_z=z)
       
        log_partition, log_weights, sample, acp_rate = hais_gauss(pot_target=pot_fun, num_chains=100, input_batch_size=mb_demo_size, dim=z_dim,
                                                              num_scheduled_dists=1000,
                                                              num_leaps=5,
                                                              step_size=0.25)
        

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        #saver.restore(sess, model_path+"vae_dcnn_relu-hei-mnist-alpha1e+00-zd32-hd500-mbs128-mbn95000-h30-l5-reg1e-06-dybin-lr1e-04.ckpt-50000")
        saver.restore(sess,tf.train.latest_checkpoint(model_path))
        print("Model restored.")
        
        log_parts=[]
        for i in range(100):
            X_mb_demo, _ = dataset.test.next_batch(mb_demo_size)
            log_part,acp = sess.run([log_partition,acp_rate], feed_dict={X_batch_demo: X_mb_demo})
            print('iter: {}, average-log-partition: {}, average-acp-rate: {}'.format(i+1,np.mean(log_part),np.mean(acp)))
            log_parts.append(np.mean(log_part))

        return log_parts
        

        


if __name__ == '__main__':
    mnist = load_iwae_binarised_mnist_dataset()
    log_partition_list = demo(dataset=mnist,device='CPU')
    print('test set log(x): {}'.format(np.mean(np.asarray(log_partition_list))))
