import tensorflow as tf
import numpy as np

from core.ham import leapfrog

def demo_jacobian(x_new, r_new, x, r):
    dx_newdx = tf.reduce_sum(tf.gradients(x_new, x))
    dr_newdr = tf.reduce_sum(tf.gradients(r_new, r))
    dx_newdr = tf.reduce_sum(tf.gradients(x_new, r))
    dr_newdx = tf.reduce_sum(tf.gradients(r_new, x))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([x_new, r_new]))
        jacob_mat = sess.run([[dx_newdx, dx_newdr],
                              [dr_newdx, dr_newdr]])
        jacob_mat_det = np.linalg.det(jacob_mat)
        print('Jacobian matrix: {}'.format(jacob_mat))
        print('Jacobian determinant: {}'.format(jacob_mat_det))


if __name__ == '__main__':
    x = tf.constant(name='position', value=np.random.normal(size=[1]))
    r = tf.constant(name='momentum', value=np.random.normal(size=[1]))
    pot_fun = lambda x: 0.5*tf.reduce_sum(x**2)/5
    x_new, r_new = leapfrog(x, r, pot_fun, eps=.2, numleap=20, r_var=1.0, back_prop=True, stop_gradient_pot=False)
    x_new_sg, r_new_sg = leapfrog(x, r, pot_fun, eps=.2, numleap=20, r_var=1.0, back_prop=True, stop_gradient_pot=True)
    print('check the jacobian matrix and determinant:')
    print('without stoping gradient:')
    demo_jacobian(x_new, r_new, x, r)
    print('with stoping gradient:')
    demo_jacobian(x_new_sg, r_new_sg, x, r)


