import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
## reference code Zhou, P. ,  Hou, Y. , &  Feng, J. . (2018). Deep Adversarial Subspace Clustering. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE.

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

# class Discriminator(object):
#     def __init__(self, x_dim=8007):
#         self.x_dim = x_dim
#         self.name = '10x_73k/clus_wgan/d_net'
#
#     def __call__(self, x, reuse=True):
#         # 变量命名空间，共享变量名？
#         with tf.variable_scope(self.name) as vs:
#             if reuse:
#                 vs.reuse_variables()
#             # 生成一个全连接层；输入为x,输出下一层单元的个数为2565
#             fc1 = tc.layers.fully_connected(
#                 x, 256,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 activation_fn=tf.identity
#             )
#             fc1 = leaky_relu(fc1)
#
#             fc2 = tc.layers.fully_connected(
#                 fc1, 256,
#                 weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                 activation_fn=tf.identity
#             )
#             fc2 = leaky_relu(fc2)
#
#             fc3 = tc.layers.fully_connected(fc2, 1,
#                                             weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                                             activation_fn=tf.identity
#                                             )
#             return fc3
#
#     @property
#     def vars(self):
#         return [var for var in tf.global_variables() if self.name in var.name]


class Decoder(object):
    def __init__(self, z_dim=44, x_dim=8007):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = '10x_73k/wgan/decoder'

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = tcl.fully_connected(
                z, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tcl.fully_connected(
                fc1, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc2 = leaky_relu(fc2)

            fc3 = tc.layers.fully_connected(
                fc2, self.x_dim,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class Encoder(object):
    def __init__(self, z_dim=44, x_dim=8007):
        self.z_dim = z_dim
        # self.dim_gen = dim_gen
        self.x_dim = x_dim
        self.name = '10x_73k/wgan/encoder'

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            fc1 = tc.layers.fully_connected(
                x, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)

            fc2 = tc.layers.fully_connected(
                fc1, 256,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc2 = leaky_relu(fc2)

            fc3 = tc.layers.fully_connected(fc2, self.z_dim, activation_fn=tf.identity)
            # logits = fc3[:, self.dim_gen:]
            # y = tf.nn.softmax(logits)
            return fc3

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
